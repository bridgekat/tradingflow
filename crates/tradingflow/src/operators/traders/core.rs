//! Port aliases shared by every trader, plus the realistic-execution book and
//! tick loop shared by [`SimpleTrader`](super::SimpleTrader) and
//! [`RandomTrader`](super::RandomTrader). Private: nothing here is re-exported.

use crate::data::{Array, ArrayView};
use crate::ports::{ArrayPort, ClockPort};

/// A `[num_stocks]` state-array port.
pub(super) type Vp = ArrayPort<f64, 1>;

/// An event stream input: a clock paired with a `[num_stocks]` value array.
pub(super) type Sp = (ClockPort, Vp);

/// The trader inputs `((positions), (close), adjusts, upper, lower)`: the
/// positions and close legs are event streams (their clocks mark a new target
/// and an executable close batch respectively); adjusts, upper and lower are
/// plain arrays read at execution time (the cumulative adjust factor is safe
/// to re-read — the ratio against `last_adjust` makes stale repeats no-ops).
pub(super) type TraderInputs = (Sp, Sp, Vp, Vp, Vp);

/// The trader input views.
pub(super) type TraderValues<'a> = (
    (ArrayView<'a, bool, 0>, ArrayView<'a, f64, 1>),
    (ArrayView<'a, bool, 0>, ArrayView<'a, f64, 1>),
    ArrayView<'a, f64, 1>,
    ArrayView<'a, f64, 1>,
    ArrayView<'a, f64, 1>,
);

/// Shared realistic-execution state for [`SimpleTrader`](super::SimpleTrader) /
/// [`RandomTrader`](super::RandomTrader).
pub(super) struct TraderCore {
    num_stocks: usize,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
    cash: f64,
    shares: Vec<f64>,
    last_adjust: Vec<f64>,
    last_close: Vec<f64>,
    pending: Option<Vec<f64>>,
    lots_buf: Vec<f64>,
    pub(super) out: Array<f64, 1>,
}

impl TraderCore {
    pub(super) fn new(
        num_stocks: usize,
        initial_cash: f64,
        lot_size: f64,
        fee_base: f64,
        fee_rate: f64,
    ) -> Self {
        let n = num_stocks;
        Self {
            num_stocks: n,
            lot_size,
            fee_base,
            fee_rate,
            cash: initial_cash,
            shares: vec![0.0; n],
            last_adjust: vec![1.0; n],
            last_close: vec![f64::NAN; n],
            pending: None,
            lots_buf: vec![0.0; n],
            out: Array::zeros([2]),
        }
    }
}

/// One realistic-execution tick, shared by both traders. `cols` are the five
/// materialized `[positions, close, adjusts, upper, lower]` slices;
/// `pos_eventful` is the positions clock (a new target arrived); `can_exec`
/// is the close clock (a pending rebalance is only executed against a fresh
/// close batch — on a generation with no close pulse there is no market to
/// trade in, and the pending signal is held). `lots` computes the per-stock
/// net trade in *lots*. Writes `[holdings, cash]` into `s.out`.
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::needless_range_loop,
    reason = "i walks the per-stock book arrays (shares/last_close/last_adjust) in lockstep with the price inputs"
)]
fn run_tick<L>(
    s: &mut TraderCore,
    pos_eventful: bool,
    can_exec: bool,
    cols: [&[f64]; 5],
    mut lots: L,
) where
    L: FnMut(f64, &[f64], &[f64], f64, &[f64], &mut [f64]),
{
    let [soft, closes, adjusts, upper, lower] = cols;
    let n = s.num_stocks;

    // Reinvest dividends.
    for i in 0..n {
        let a = adjusts[i];
        if a.is_finite() && a > 0.0 {
            if s.last_adjust[i] > 0.0 {
                s.shares[i] *= a / s.last_adjust[i];
            }
            s.last_adjust[i] = a;
        }
    }
    // Last-valid-close carry-forward.
    for i in 0..n {
        if closes[i].is_finite() {
            s.last_close[i] = closes[i];
        }
    }

    if can_exec && let Some(pending) = s.pending.take() {
        // Step 1: force-liquidate suspended holdings at last_close, each charged
        // a fee like a normal trade.
        for i in 0..n {
            let valid_exec = closes[i].is_finite() && closes[i] > 0.0;
            if s.shares[i] != 0.0 && !valid_exec && s.last_close[i].is_finite() {
                let sell_value = s.shares[i] * s.last_close[i];
                let fee = s.fee_base.max(sell_value.abs() * s.fee_rate);
                s.cash += sell_value - fee;
                s.shares[i] = 0.0;
            }
        }
        // Step 2: portfolio value after force-liquidation.
        let mut current_value = s.cash;
        for i in 0..n {
            if s.shares[i] != 0.0 && s.last_close[i].is_finite() {
                current_value += s.shares[i] * s.last_close[i];
            }
        }
        // Step 3: per-stock net lots from the operator-specific sizer.
        lots(
            current_value,
            closes,
            &s.shares,
            s.lot_size,
            &pending,
            &mut s.lots_buf,
        );
        // Step 4: execute lots at today's close.
        for i in 0..n {
            let valid_exec = closes[i].is_finite() && closes[i] > 0.0;
            if !valid_exec {
                continue;
            }
            let p = closes[i];
            let mut trade_shares = s.lots_buf[i] * s.lot_size;
            if (s.shares[i] + trade_shares).abs() < s.lot_size {
                trade_shares = -s.shares[i];
            }
            if trade_shares > 0.0 && upper[i].is_finite() && p >= upper[i] {
                continue;
            }
            if trade_shares < 0.0 && lower[i].is_finite() && p <= lower[i] {
                continue;
            }
            if trade_shares != 0.0 {
                let trade_value = trade_shares * p;
                let fee = s.fee_base.max(trade_value.abs() * s.fee_rate);
                s.cash -= trade_value + fee;
                s.shares[i] += trade_shares;
            }
        }
    }

    if pos_eventful {
        s.pending = Some(soft.to_vec());
    }

    let mut holdings_value = 0.0;
    for i in 0..n {
        if s.shares[i] != 0.0 && s.last_close[i].is_finite() {
            holdings_value += s.shares[i] * s.last_close[i];
        }
    }
    // Ruin check — see the note in `Benchmark::compute`: `!(x > 0.0)` catches
    // `NaN` (a missing price makes net worth `NaN`), which `x <= 0.0` does not.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(s.cash + holdings_value > 0.0) {
        s.shares.iter_mut().for_each(|x| *x = 0.0);
        s.cash = 0.0;
        s.pending = None;
        holdings_value = 0.0;
    }
    let out = s.out.data_mut();
    out[0] = holdings_value;
    out[1] = s.cash;
}

/// Materialize the five trader inputs into contiguous slices, run `lots`, and
/// return the `[holdings_value, cash]` state view.
pub(super) fn run_trader<'a, 'b: 'a, L>(
    core: &'b mut TraderCore,
    values: TraderValues<'a>,
    lots: L,
) -> ArrayView<'a, f64, 1>
where
    L: FnMut(f64, &[f64], &[f64], f64, &[f64], &mut [f64]),
{
    let ((pos_eventful, pos), (can_exec, close), adj, up, lo) = values;
    let (a, b, c, d, e) = (
        pos.to_contiguous(),
        close.to_contiguous(),
        adj.to_contiguous(),
        up.to_contiguous(),
        lo.to_contiguous(),
    );
    run_tick(core, *pos_eventful, *can_exec, [&a, &b, &c, &d, &e], lots);
    core.out.view()
}
