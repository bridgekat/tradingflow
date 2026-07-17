//! Trader (execution) operators, implemented directly on
//! [`Operator`].
//!
//! A trader turns a strategy's target weights into a simulated portfolio NAV: it
//! reinvests dividends, executes rebalances against close prices, marks the book
//! to market, and reports `[holdings_value, cash]` (total NAV = their sum). This
//! is the step that closes the backtest loop, so having it in Rust lets a full
//! backtest run on the native operator set alone. [`Benchmark`], the
//! realistic-cost [`SimpleTrader`], and the stochastic [`RandomTrader`] are all
//! native Rust; there are no Python trader implementations.
//!
//! Each takes the five `[num_stocks]` array views
//! `(positions, close, adjusts, upper_limit, lower_limit)` as a 5-tuple of
//! `ArrayPort<f64, 1>`, and outputs `[2]` = `[holdings_value, cash]`.
//! Only the **positions** notify flag is consulted (one-tick-delayed execution).

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::graph::typed::Operator;
use crate::ports::ArrayPort;
use crate::{Array, ArrayView, Instant};

/// A single `[num_stocks]` view port — the per-input edge of every trader.
type Vp = ArrayPort<f64, 1>;
/// The five trader inputs `(positions, close, adjusts, upper, lower)`.
type TraderInputs = (Vp, Vp, Vp, Vp, Vp);
/// The five trader input values as `(notify, view)` pairs.
type TraderValues<'a> = (
    (bool, ArrayView<'a, f64, 1>),
    (bool, ArrayView<'a, f64, 1>),
    (bool, ArrayView<'a, f64, 1>),
    (bool, ArrayView<'a, f64, 1>),
    (bool, ArrayView<'a, f64, 1>),
);

/// Frictionless benchmark executor: replicates target weights exactly, with
/// dividend reinvestment, one-tick-delayed mark-on-close execution, idealised
/// force-liquidation of suspended holdings, A-shares price-limit blocking, and a
/// bankruptcy wipe. Output is `[2]` = `[holdings_value, cash]`; NAV is their sum.
pub struct Benchmark {
    num_stocks: usize,
    initial_cash: f64,
    use_adjusts: bool,
}

impl Benchmark {
    /// `num_stocks` is the cross-section width (= the input arrays' length);
    /// `initial_cash` seeds the book (use `1.0` for a unit-NAV index);
    /// `use_adjusts` toggles dividend reinvestment via the adjust factors.
    pub fn new(num_stocks: usize, initial_cash: f64, use_adjusts: bool) -> Self {
        Self {
            num_stocks,
            initial_cash,
            use_adjusts,
        }
    }
}

/// Runtime state for [`Benchmark`]: the book (cash + per-stock shares), the
/// dividend/close carry-forwards, the rebalance signal pending one-tick
/// execution, and the `[2]` output buffer.
pub struct BenchmarkState {
    num_stocks: usize,
    use_adjusts: bool,
    cash: f64,
    shares: Vec<f64>,
    last_adjust: Vec<f64>,
    last_close: Vec<f64>,
    pending: Option<Vec<f64>>,
    out: Array<f64, 1>,
}

impl Operator for Benchmark {
    type Inputs = TraderInputs;
    type Outputs = Vp;
    type Context = Instant;
    type State = BenchmarkState;

    fn init(self) -> BenchmarkState {
        let n = self.num_stocks;
        BenchmarkState {
            num_stocks: n,
            use_adjusts: self.use_adjusts,
            cash: self.initial_cash,
            shares: vec![0.0; n],
            last_adjust: vec![1.0; n],
            last_close: vec![f64::NAN; n],
            pending: None,
            out: Array::zeros([2]),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((pos_notified, pos), (_, close), (_, adj), (_, up), (_, lo)): TraderValues<'a>,
        _: &Instant,
        state: &'b mut BenchmarkState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        if init {
            assert_eq!(
                pos.len(),
                state.num_stocks,
                "Benchmark: input length {} != num_stocks {}",
                pos.len(),
                state.num_stocks,
            );
            return (false, state.out.view());
        }
        let n = state.num_stocks;
        let positions = pos.to_contiguous();
        let closes = close.to_contiguous();
        let adjusts = adj.to_contiguous();
        let upper = up.to_contiguous();
        let lower = lo.to_contiguous();

        // Reinvest dividends: scale held shares by the change in adjust factor.
        if state.use_adjusts {
            for i in 0..n {
                let a = adjusts[i];
                if a.is_finite() && a > 0.0 {
                    if state.last_adjust[i] > 0.0 {
                        state.shares[i] *= a / state.last_adjust[i];
                    }
                    state.last_adjust[i] = a;
                }
            }
        }

        // Carry forward the last valid close for stocks that ticked this cycle;
        // suspended stocks retain their previous last-valid close. Done before
        // the rebalance so today's close is both the sizing and marking price.
        for i in 0..n {
            if closes[i].is_finite() {
                state.last_close[i] = closes[i];
            }
        }

        // Execute the rebalance signalled one tick ago, at today's close.
        if let Some(pending) = state.pending.take() {
            // Step 1: idealised force-liquidation of held stocks with no valid
            // exec price today (suspended/delisted), at their last valid close.
            for i in 0..n {
                let valid_exec = closes[i].is_finite() && closes[i] > 0.0;
                if state.shares[i] != 0.0 && !valid_exec && state.last_close[i].is_finite() {
                    state.cash += state.shares[i] * state.last_close[i];
                    state.shares[i] = 0.0;
                }
            }
            // Step 2: portfolio value after force-liquidation (last-valid close).
            let mut current_value = state.cash;
            for i in 0..n {
                if state.shares[i] != 0.0 && state.last_close[i].is_finite() {
                    current_value += state.shares[i] * state.last_close[i];
                }
            }
            // Step 3: single net trade to target at today's close, tradable
            // stocks only, with A-shares price-limit blocking.
            for i in 0..n {
                let valid_exec = closes[i].is_finite() && closes[i] > 0.0;
                if !valid_exec {
                    continue;
                }
                let close = closes[i];
                let target_shares = pending[i] * current_value / close;
                let mut trade = target_shares - state.shares[i];
                let block_buy = upper[i].is_finite() && close >= upper[i] && trade > 0.0;
                let block_sell = lower[i].is_finite() && close <= lower[i] && trade < 0.0;
                if block_buy || block_sell {
                    trade = 0.0;
                }
                state.cash -= trade * close;
                state.shares[i] += trade;
            }
        }

        // Capture a new target for execution on the NEXT tick.
        if pos_notified {
            state.pending = Some(positions.to_vec());
        }

        // Output [holdings_value, cash], with a bankruptcy wipe.
        let mut holdings_value = 0.0;
        for i in 0..n {
            if state.shares[i] != 0.0 && state.last_close[i].is_finite() {
                holdings_value += state.shares[i] * state.last_close[i];
            }
        }
        // Ruin check. The negation is deliberate and NOT equivalent to
        // `<= 0.0`: net worth is `NaN` when a held price is missing, and
        // `!(NaN > 0.0)` is `true` while `NaN <= 0.0` is `false`. Rewriting
        // this comparison would let a NaN-valued book keep trading.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(state.cash + holdings_value > 0.0) {
            state.shares.iter_mut().for_each(|s| *s = 0.0);
            state.cash = 0.0;
            state.pending = None;
            holdings_value = 0.0;
        }
        let out = state.out.data_mut();
        out[0] = holdings_value;
        out[1] = state.cash;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: TraderValues<'a>,
        _: &Instant,
        state: &'b BenchmarkState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.out.view())
    }
}

// ===========================================================================
// SimpleTrader / RandomTrader — realistic-cost execution
// ===========================================================================

/// Shared realistic-execution state for [`SimpleTrader`] / [`RandomTrader`].
struct TraderCore {
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
    out: Array<f64, 1>,
}

impl TraderCore {
    fn new(
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
/// `pos_notified` is the positions notify flag; `lots` computes the per-stock
/// net trade in *lots*. Writes `[holdings, cash]` into `s.out`.
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::needless_range_loop,
    reason = "i walks the per-stock book arrays (shares/last_close/last_adjust) in lockstep with the price inputs"
)]
fn run_tick<L>(
    s: &mut TraderCore,
    pos_notified: bool,
    cols: [&[f64]; 5],
    init: bool,
    mut lots: L,
) -> bool
where
    L: FnMut(f64, &[f64], &[f64], f64, &[f64], &mut [f64]),
{
    let [soft, closes, adjusts, upper, lower] = cols;
    if init {
        assert_eq!(
            soft.len(),
            s.num_stocks,
            "trader: input length {} != num_stocks {}",
            soft.len(),
            s.num_stocks,
        );
        return false;
    }
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

    if let Some(pending) = s.pending.take() {
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

    if pos_notified {
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
    true
}

/// Materialize the five trader inputs into contiguous slices, run `lots`, and
/// return `(positions_notify, holdings_view)`.
fn run_trader<'a, 'b: 'a, L>(
    core: &'b mut TraderCore,
    values: TraderValues<'a>,
    init: bool,
    lots: L,
) -> (bool, ArrayView<'a, f64, 1>)
where
    L: FnMut(f64, &[f64], &[f64], f64, &[f64], &mut [f64]),
{
    let ((pn, pos), (_, close), (_, adj), (_, up), (_, lo)) = values;
    let (a, b, c, d, e) = (
        pos.to_contiguous(),
        close.to_contiguous(),
        adj.to_contiguous(),
        up.to_contiguous(),
        lo.to_contiguous(),
    );
    let notify = run_tick(core, pn, [&a, &b, &c, &d, &e], init, lots);
    (notify, core.out.view())
}

/// Value-weight lot sizing (the [`SimpleTrader`] rule).
fn value_weight_lots(
    current_value: f64,
    exec: &[f64],
    shares: &[f64],
    lot_size: f64,
    soft: &[f64],
    out: &mut [f64],
) {
    for i in 0..out.len() {
        let p = exec[i];
        if !p.is_finite() || p <= 0.0 {
            out[i] = 0.0;
            continue;
        }
        let target_shares = soft[i] * current_value / p;
        out[i] = ((target_shares - shares[i]) / lot_size).round();
    }
}

/// Random lot sizing ([`RandomTrader`]): Efraimidis–Spirakis weighted reservoir.
#[allow(clippy::too_many_arguments)]
fn random_lots(
    rng: &mut StdRng,
    portfolio_size: usize,
    current_value: f64,
    exec: &[f64],
    shares: &[f64],
    lot_size: f64,
    soft: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    out.iter_mut().for_each(|x| *x = 0.0);

    let mut weights = vec![0.0f64; n];
    let mut total = 0.0;
    for i in 0..n {
        if exec[i].is_finite() && exec[i] > 0.0 {
            let w = soft[i].max(0.0);
            weights[i] = w;
            total += w;
        }
    }
    if total <= 0.0 {
        return;
    }
    let n_candidates = weights.iter().filter(|&&w| w > 0.0).count();
    let n_select = portfolio_size.min(n_candidates);
    if n_select == 0 {
        return;
    }

    let mut keyed: Vec<(f64, usize)> = (0..n)
        .filter(|&i| weights[i] > 0.0)
        .map(|i| {
            let u: f64 = rng.random_range(0.0..1.0);
            let u = if u <= 0.0 { f64::MIN_POSITIVE } else { u };
            (u.powf(1.0 / weights[i]), i)
        })
        .collect();
    keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let hard = total / portfolio_size as f64;
    let mut hard_w = vec![0.0f64; n];
    for &(_, i) in keyed.iter().take(n_select) {
        hard_w[i] = hard;
    }
    for i in 0..n {
        let p = exec[i];
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let target_shares = hard_w[i] * current_value / p;
        out[i] = ((target_shares - shares[i]) / lot_size).round();
    }
}

/// Realistic-cost executor with **value-weight** sizing.
pub struct SimpleTrader {
    num_stocks: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
}

impl SimpleTrader {
    pub fn new(
        num_stocks: usize,
        initial_cash: f64,
        lot_size: f64,
        fee_base: f64,
        fee_rate: f64,
    ) -> Self {
        Self {
            num_stocks,
            initial_cash,
            lot_size,
            fee_base,
            fee_rate,
        }
    }
}

/// Runtime state for [`SimpleTrader`].
pub struct SimpleTraderState {
    core: TraderCore,
}

impl Operator for SimpleTrader {
    type Inputs = TraderInputs;
    type Outputs = Vp;
    type Context = Instant;
    type State = SimpleTraderState;

    fn init(self) -> SimpleTraderState {
        SimpleTraderState {
            core: TraderCore::new(
                self.num_stocks,
                self.initial_cash,
                self.lot_size,
                self.fee_base,
                self.fee_rate,
            ),
        }
    }

    fn compute<'a, 'b: 'a>(
        values: TraderValues<'a>,
        _: &Instant,
        state: &'b mut SimpleTraderState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        run_trader(&mut state.core, values, init, value_weight_lots)
    }

    fn passthrough<'a, 'b: 'a>(
        _: TraderValues<'a>,
        _: &Instant,
        state: &'b SimpleTraderState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.core.out.view())
    }
}

/// Realistic-cost executor with **random** sizing.
pub struct RandomTrader {
    num_stocks: usize,
    portfolio_size: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
    seed: u64,
}

impl RandomTrader {
    pub fn new(
        num_stocks: usize,
        portfolio_size: usize,
        initial_cash: f64,
        lot_size: f64,
        fee_base: f64,
        fee_rate: f64,
        seed: u64,
    ) -> Self {
        Self {
            num_stocks,
            portfolio_size,
            initial_cash,
            lot_size,
            fee_base,
            fee_rate,
            seed,
        }
    }
}

/// Runtime state for [`RandomTrader`]: the shared book plus the seeded RNG.
pub struct RandomTraderState {
    core: TraderCore,
    rng: StdRng,
    portfolio_size: usize,
}

impl Operator for RandomTrader {
    type Inputs = TraderInputs;
    type Outputs = Vp;
    type Context = Instant;
    type State = RandomTraderState;

    fn init(self) -> RandomTraderState {
        RandomTraderState {
            core: TraderCore::new(
                self.num_stocks,
                self.initial_cash,
                self.lot_size,
                self.fee_base,
                self.fee_rate,
            ),
            rng: StdRng::seed_from_u64(self.seed),
            portfolio_size: self.portfolio_size,
        }
    }

    fn compute<'a, 'b: 'a>(
        values: TraderValues<'a>,
        _: &Instant,
        state: &'b mut RandomTraderState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        let RandomTraderState {
            core,
            rng,
            portfolio_size,
        } = state;
        let ps = *portfolio_size;
        run_trader(core, values, init, |cv, exec, shares, ls, soft, out| {
            random_lots(rng, ps, cv, exec, shares, ls, soft, out)
        })
    }

    fn passthrough<'a, 'b: 'a>(
        _: TraderValues<'a>,
        _: &Instant,
        state: &'b RandomTraderState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.core.out.view())
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Frictionless benchmark executor: replicates target weights exactly.
pub fn benchmark(num_stocks: usize, initial_cash: f64, use_adjusts: bool) -> Benchmark {
    Benchmark::new(num_stocks, initial_cash, use_adjusts)
}

/// Realistic-cost executor: lot rounding plus a base + rate fee model.
pub fn simple_trader(
    num_stocks: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
) -> SimpleTrader {
    SimpleTrader::new(num_stocks, initial_cash, lot_size, fee_base, fee_rate)
}

/// Stochastic executor: holds a seeded random `portfolio_size` subset.
pub fn random_trader(
    num_stocks: usize,
    portfolio_size: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
    seed: u64,
) -> RandomTrader {
    RandomTrader::new(
        num_stocks,
        portfolio_size,
        initial_cash,
        lot_size,
        fee_base,
        fee_rate,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::Pool;
    use crate::graph::typed::{Builder, NodeHandle, PortHandle, ViewSource};
    use crate::operators::constant::array_cell;
    use crate::ports::ArrayValue;

    fn arr(v: &[f64]) -> Array<f64, 1> {
        Array::from_vec([v.len()], v.to_vec())
    }

    /// Push a rank-1 array [`array_cell`] of `v`; return the source handle
    /// (for `state_mut`) and its `ArrayPort` view handle (for wiring).
    fn src(
        b: &mut Builder<Instant>,
        v: &[f64],
    ) -> (
        NodeHandle<ViewSource<ArrayValue<f64, 1>, Instant>>,
        PortHandle<Vp>,
    ) {
        b.source(array_cell(arr(v)))
    }

    #[test]
    fn benchmark_one_tick_delay_exec_and_mark() {
        let nan = f64::NAN;
        let mut b = Builder::new(Instant::MIN);
        let (pos, posv) = src(&mut b, &[nan, nan]);
        let (close, closev) = src(&mut b, &[nan, nan]);
        let (_adj, adjv) = src(&mut b, &[1.0, 1.0]);
        let (_up, upv) = src(&mut b, &[nan, nan]);
        let (_lo, lov) = src(&mut b, &[nan, nan]);
        let out = b.segment(Benchmark::new(2, 1.0, true), (posv, closev, adjv, upv, lov));
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(pos) = arr(&[0.5, 0.5]);
        *g.state_mut(close) = arr(&[10.0, 20.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.view(out).as_slice().unwrap(), &[0.0, 1.0]);

        *g.state_mut(close) = arr(&[11.0, 22.0]);
        g.stabilize(&mut pool);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!((o[0] - 1.0).abs() < 1e-12, "tick2 holdings {} != 1.0", o[0]);
        assert!(o[1].abs() < 1e-12, "tick2 cash {} != 0", o[1]);

        *g.state_mut(close) = arr(&[12.0, 22.0]);
        g.stabilize(&mut pool);
        let o = g.view(out).as_slice().unwrap().to_vec();
        let nav = o[0] + o[1];
        let expected = 0.5 * (12.0 / 11.0) + 0.5;
        assert!(
            (nav - expected).abs() < 1e-12,
            "tick3 NAV {} != {}",
            nav,
            expected
        );
    }

    #[test]
    fn benchmark_reinvests_dividends() {
        let nan = f64::NAN;
        let mut b = Builder::new(Instant::MIN);
        let (pos, posv) = src(&mut b, &[nan]);
        let (close, closev) = src(&mut b, &[nan]);
        let (adj, adjv) = src(&mut b, &[1.0]);
        let (_up, upv) = src(&mut b, &[nan]);
        let (_lo, lov) = src(&mut b, &[nan]);
        let out = b.segment(Benchmark::new(1, 1.0, true), (posv, closev, adjv, upv, lov));
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(pos) = arr(&[1.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!(
            (o[0] - 1.0).abs() < 1e-12 && o[1].abs() < 1e-12,
            "fully invested: {o:?}"
        );

        *g.state_mut(adj) = arr(&[2.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!(
            (o[0] - 2.0).abs() < 1e-12,
            "reinvested holdings {} != 2.0",
            o[0]
        );
    }

    #[test]
    fn simple_trader_value_weight_with_fees_and_lots() {
        let nan = f64::NAN;
        let mut b = Builder::new(Instant::MIN);
        let (pos, posv) = src(&mut b, &[nan]);
        let (close, closev) = src(&mut b, &[nan]);
        let (_adj, adjv) = src(&mut b, &[1.0]);
        let (_up, upv) = src(&mut b, &[nan]);
        let (_lo, lov) = src(&mut b, &[nan]);
        let out = b.segment(
            SimpleTrader::new(1, 1_000_000.0, 100.0, 5.0, 0.001),
            (posv, closev, adjv, upv, lov),
        );
        let mut g = b.build();
        let mut pool = Pool::new(0);

        *g.state_mut(pos) = arr(&[1.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.view(out).as_slice().unwrap(), &[0.0, 1_000_000.0]);

        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.view(out).as_slice().unwrap(), &[1_000_000.0, -1000.0]);
    }

    #[test]
    fn random_trader_invests_and_is_seed_deterministic() {
        let nan = f64::NAN;
        let run = || {
            let mut b = Builder::new(Instant::MIN);
            let (pos, posv) = src(&mut b, &[nan; 5]);
            let (close, closev) = src(&mut b, &[nan; 5]);
            let (_adj, adjv) = src(&mut b, &[1.0; 5]);
            let (_up, upv) = src(&mut b, &[nan; 5]);
            let (_lo, lov) = src(&mut b, &[nan; 5]);
            let out = b.segment(
                RandomTrader::new(5, 2, 1000.0, 1.0, 0.0, 0.0, 0),
                (posv, closev, adjv, upv, lov),
            );
            let mut g = b.build();
            let mut pool = Pool::new(0);
            *g.state_mut(pos) = arr(&[0.2; 5]);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool);
            let invested = g.view(out).as_slice().unwrap().to_vec();
            *g.state_mut(close) = arr(&[11.0, 9.0, 10.0, 10.0, 10.0]);
            g.stabilize(&mut pool);
            let marked = g.view(out).as_slice().unwrap().to_vec();
            (invested, marked)
        };
        let (inv1, mk1) = run();
        let (_inv2, mk2) = run();

        assert_eq!(
            inv1,
            vec![1000.0, 0.0],
            "should be fully invested after rebalance"
        );
        assert_eq!(mk1, mk2, "same seed must give identical results");
        let nav = mk1[0] + mk1[1];
        assert!(
            nav > 900.0 && nav < 1100.0,
            "tick3 NAV {nav} out of held-leg bounds"
        );
    }
}
