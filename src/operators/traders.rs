//! Trader (execution) operators — native Rust ports of the `flowops` traders.
//!
//! A trader turns a strategy's target weights into a simulated portfolio NAV: it
//! reinvests dividends, executes rebalances against close prices, marks the book
//! to market, and reports `[holdings_value, cash]` (total NAV = their sum). This
//! is the step that closes the backtest loop, so having it in Rust lets a full
//! backtest run on the native operator set alone — the Python `flowops` traders
//! (the realistic-cost `SimpleTrader` and the stochastic `RandomTrader`) remain
//! as optional strategy building blocks.
//!
//! [`Benchmark`] is the frictionless ideal executor (no fees, no lot rounding,
//! instant fills) used both as a strategy's NAV simulator and to build index
//! baselines. Ported verbatim from `flowops.traders.benchmark`; the only
//! numeric difference is summation order (the NumPy original uses pairwise
//! `np.sum`; this uses sequential sums), so NAVs match to rounding, not bit-for-bit.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use flowgraph::typed::{Operator, RefPort, RefPorts};

use crate::Array;

/// Frictionless benchmark executor: replicates target weights exactly, with
/// dividend reinvestment, one-tick-delayed mark-on-close execution, idealised
/// force-liquidation of suspended holdings, A-shares price-limit blocking, and a
/// bankruptcy wipe.
///
/// Inputs are a [`RefPorts<Array<f64>>`](flowgraph::typed::RefPorts) of exactly
/// five `[num_stocks]` arrays, in order:
/// `[positions, close, adjusts, upper_limit, lower_limit]`. Only the **positions**
/// notify flag is consulted: a new target is captured on the tick its weights
/// fire and executed at the *next* close (the one-tick delay keeps the signal's
/// information set strictly older than the execution price). Output is `[2]` =
/// `[holdings_value, cash]`; total NAV is their sum.
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
    out: Array<f64>,
}

impl Operator for Benchmark {
    type Inputs = RefPorts<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
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
            out: Array::zeros(&[2]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (flags, values): (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b mut BenchmarkState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            assert_eq!(
                values.len(),
                5,
                "Benchmark expects 5 inputs [positions, close, adjusts, upper, lower]",
            );
            assert_eq!(
                values[0].stride(),
                state.num_stocks,
                "Benchmark: input length {} != num_stocks {}",
                values[0].stride(),
                state.num_stocks,
            );
            return (false, &state.out);
        }
        let n = state.num_stocks;
        let positions = values[0].as_slice();
        let closes = values[1].as_slice();
        let adjusts = values[2].as_slice();
        let upper = values[3].as_slice();
        let lower = values[4].as_slice();

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
            // stocks only, with A-shares price-limit blocking (no buys at
            // limit-up, no sells at limit-down; NaN limit = no constraint).
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
        if flags[0] {
            state.pending = Some(positions.to_vec());
        }

        // Output [holdings_value, cash], with a bankruptcy wipe: if total NAV is
        // non-positive (e.g. a long-short book whose loss exceeds equity), zero
        // everything — an absorbing wiped-out state, as a margin call would.
        let mut holdings_value = 0.0;
        for i in 0..n {
            if state.shares[i] != 0.0 && state.last_close[i].is_finite() {
                holdings_value += state.shares[i] * state.last_close[i];
            }
        }
        if !(state.cash + holdings_value > 0.0) {
            state.shares.iter_mut().for_each(|s| *s = 0.0);
            state.cash = 0.0;
            state.pending = None;
            holdings_value = 0.0;
        }
        let out = state.out.as_mut_slice();
        out[0] = holdings_value;
        out[1] = state.cash;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b BenchmarkState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
    }
}

// ===========================================================================
// SimpleTrader / RandomTrader — realistic-cost execution
// ===========================================================================

/// Shared realistic-execution state for [`SimpleTrader`] / [`RandomTrader`]:
/// the book plus the fee/lot config and scratch buffers. The only thing that
/// differs between the two is how target lots are sized (see `run_tick`'s
/// `lots` callback).
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
    out: Array<f64>,
}

impl TraderCore {
    fn new(num_stocks: usize, initial_cash: f64, lot_size: f64, fee_base: f64, fee_rate: f64) -> Self {
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
            out: Array::zeros(&[2]),
        }
    }
}

/// One realistic-execution tick, shared by both traders. Inputs are
/// `[positions, close, adjusts, upper, lower]`; `lots` computes the per-stock
/// net trade in *lots* given `(current_value, exec_price, shares, lot_size,
/// target_positions, out_lots)` (the only operator-specific step). Returns the
/// notify flag (always `true` off the build call). Writes `[holdings, cash]`
/// into `s.out`. Mirrors `flowops.traders._base.SimpleTrader.compute`.
fn run_tick<L>(
    s: &mut TraderCore,
    flags: &[bool],
    values: &[&Array<f64>],
    init: bool,
    mut lots: L,
) -> bool
where
    L: FnMut(f64, &[f64], &[f64], f64, &[f64], &mut [f64]),
{
    if init {
        assert_eq!(
            values.len(),
            5,
            "trader expects 5 inputs [positions, close, adjusts, upper, lower]",
        );
        assert_eq!(
            values[0].stride(),
            s.num_stocks,
            "trader: input length {} != num_stocks {}",
            values[0].stride(),
            s.num_stocks,
        );
        return false;
    }
    let n = s.num_stocks;
    let soft = values[0].as_slice();
    let closes = values[1].as_slice();
    let adjusts = values[2].as_slice();
    let upper = values[3].as_slice();
    let lower = values[4].as_slice();

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
        lots(current_value, closes, &s.shares, s.lot_size, &pending, &mut s.lots_buf);
        // Step 4: execute lots at today's close — sub-lot-remnant liquidation,
        // A-shares price-limit blocking, per-trade fees.
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

    if flags[0] {
        s.pending = Some(soft.to_vec());
    }

    let mut holdings_value = 0.0;
    for i in 0..n {
        if s.shares[i] != 0.0 && s.last_close[i].is_finite() {
            holdings_value += s.shares[i] * s.last_close[i];
        }
    }
    if !(s.cash + holdings_value > 0.0) {
        s.shares.iter_mut().for_each(|x| *x = 0.0);
        s.cash = 0.0;
        s.pending = None;
        holdings_value = 0.0;
    }
    let out = s.out.as_mut_slice();
    out[0] = holdings_value;
    out[1] = s.cash;
    true
}

/// Value-weight lot sizing: treat each target position as a market-value
/// fraction and size to it at the execution price (the [`SimpleTrader`] rule).
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

/// Random lot sizing ([`RandomTrader`]): pick `portfolio_size` tradable stocks
/// without replacement, weighted by the (clamped) target positions, equal-weight
/// them at `total_weight / portfolio_size`, and size to that at the execution
/// price. Non-chosen tradable holdings are sized to zero (sold). Uses
/// Efraimidis–Spirakis weighted reservoir keys (`u^(1/wᵢ)`, take the largest) —
/// logic-equivalent to NumPy's weighted `choice(replace=False)`, not bit-for-bit.
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

    // Sampling weights: clamped target positions on tradable stocks.
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

    // Weighted sampling without replacement: key_i = u^(1/w_i); the n_select
    // largest keys are the sample (selection is invariant to weight scaling).
    let mut keyed: Vec<(f64, usize)> = (0..n)
        .filter(|&i| weights[i] > 0.0)
        .map(|i| {
            let u: f64 = rng.gen_range(0.0..1.0);
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
    // Lots for every tradable stock: chosen → buy to `hard`, others → sell.
    for i in 0..n {
        let p = exec[i];
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let target_shares = hard_w[i] * current_value / p;
        out[i] = ((target_shares - shares[i]) / lot_size).round();
    }
}

/// Realistic-cost executor with **value-weight** sizing: treats the target
/// positions as market-value weights and trades to them with per-trade fees,
/// lot rounding, one-tick-delayed mark-on-close execution, dividend
/// reinvestment, suspended-holding force-liquidation, and A-shares price limits.
///
/// Inputs / output match [`Benchmark`]: `RefPorts<Array<f64>>` =
/// `[positions, close, adjusts, upper, lower]` → `[holdings_value, cash]`.
pub struct SimpleTrader {
    num_stocks: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
}

impl SimpleTrader {
    pub fn new(num_stocks: usize, initial_cash: f64, lot_size: f64, fee_base: f64, fee_rate: f64) -> Self {
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
    type Inputs = RefPorts<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
    type State = SimpleTraderState;

    fn init(self) -> SimpleTraderState {
        SimpleTraderState {
            core: TraderCore::new(self.num_stocks, self.initial_cash, self.lot_size, self.fee_base, self.fee_rate),
        }
    }

    fn compute<'a, 'b: 'a>(
        (flags, values): (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b mut SimpleTraderState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        let notify = run_tick(&mut state.core, flags, values, init, value_weight_lots);
        (notify, &state.core.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b SimpleTraderState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.core.out)
    }
}

/// Realistic-cost executor with **random** sizing: on each rebalance it picks
/// `portfolio_size` stocks (weighted by the target positions, without
/// replacement) and equal-weights them, sharing [`SimpleTrader`]'s execution
/// path (fees, lots, limits, …). Seeded for reproducibility, but the RNG is
/// Rust's `StdRng` — results are logic-equivalent to, not bit-identical with,
/// the NumPy original.
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
    type Inputs = RefPorts<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
    type State = RandomTraderState;

    fn init(self) -> RandomTraderState {
        RandomTraderState {
            core: TraderCore::new(self.num_stocks, self.initial_cash, self.lot_size, self.fee_base, self.fee_rate),
            rng: StdRng::seed_from_u64(self.seed),
            portfolio_size: self.portfolio_size,
        }
    }

    fn compute<'a, 'b: 'a>(
        (flags, values): (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b mut RandomTraderState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        let RandomTraderState { core, rng, portfolio_size } = state;
        let ps = *portfolio_size;
        let notify = run_tick(core, flags, values, init, |cv, exec, shares, ls, soft, out| {
            random_lots(rng, ps, cv, exec, shares, ls, soft, out)
        });
        (notify, &core.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [&'a Array<f64>]),
        state: &'b RandomTraderState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.core.out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowgraph::core::Pool;
    use flowgraph::typed::{Graph, GraphBuilder, RefSource};

    fn arr(v: &[f64]) -> Array<f64> {
        Array::from_vec(&[v.len()], v.to_vec())
    }

    /// The benchmark defers a rebalance one tick (signal at close[t] → execution
    /// at close[t+1]), then marks the book to market. Frictionless, so sizing
    /// and marking at the same close preserve NAV exactly; a later price move is
    /// reflected one-to-one.
    #[test]
    fn benchmark_one_tick_delay_exec_and_mark() {
        let nan = f64::NAN;
        let mut b = GraphBuilder::new();
        let pos = b.push_source(RefSource::new(arr(&[nan, nan])));
        let close = b.push_source(RefSource::new(arr(&[nan, nan])));
        let adj = b.push_source(RefSource::new(arr(&[1.0, 1.0])));
        let up = b.push_source(RefSource::new(arr(&[nan, nan])));
        let lo = b.push_source(RefSource::new(arr(&[nan, nan])));
        let out = b.push(Benchmark::new(2, 1.0, true), &[*pos, *close, *adj, *up, *lo][..]);
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Tick 1: target weights [0.5, 0.5] fire at close [10, 20].
        // One-tick delay: NO trade this tick — all cash, zero holdings.
        *g.state_mut(pos) = arr(&[0.5, 0.5]);
        *g.state_mut(close) = arr(&[10.0, 20.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[0.0, 1.0]);

        // Tick 2: only close ticks [11, 22] → execute the pending target at this
        // close. Frictionless ⇒ fully invested at preserved NAV (holdings≈1, cash≈0).
        *g.state_mut(close) = arr(&[11.0, 22.0]);
        g.stabilize(&mut pool);
        let o = g.ref_view(out).as_slice().to_vec();
        assert!((o[0] - 1.0).abs() < 1e-12, "tick2 holdings {} != 1.0", o[0]);
        assert!(o[1].abs() < 1e-12, "tick2 cash {} != 0", o[1]);

        // Tick 3: stock 0 +10% to 12 (stock 1 flat). No rebalance; NAV marks up
        // by the held weight on stock 0.
        *g.state_mut(close) = arr(&[12.0, 22.0]);
        g.stabilize(&mut pool);
        let o = g.ref_view(out).as_slice().to_vec();
        let nav = o[0] + o[1];
        let expected = 0.5 * (12.0 / 11.0) + 0.5;
        assert!((nav - expected).abs() < 1e-12, "tick3 NAV {} != {}", nav, expected);
    }

    /// Dividend reinvestment: an adjust-factor step scales held shares so the
    /// holding value rises by the same factor (no cash change).
    #[test]
    fn benchmark_reinvests_dividends() {
        let nan = f64::NAN;
        let mut b = GraphBuilder::new();
        let pos = b.push_source(RefSource::new(arr(&[nan])));
        let close = b.push_source(RefSource::new(arr(&[nan])));
        let adj = b.push_source(RefSource::new(arr(&[1.0])));
        let up = b.push_source(RefSource::new(arr(&[nan])));
        let lo = b.push_source(RefSource::new(arr(&[nan])));
        let out = b.push(Benchmark::new(1, 1.0, true), &[*pos, *close, *adj, *up, *lo][..]);
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Fully invest in the single stock (signal tick, then exec tick).
        *g.state_mut(pos) = arr(&[1.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        let o = g.ref_view(out).as_slice().to_vec();
        assert!((o[0] - 1.0).abs() < 1e-12 && o[1].abs() < 1e-12, "fully invested: {o:?}");

        // Adjust factor doubles (a dividend reinvested): shares double, so at the
        // same close the holding value doubles → NAV ≈ 2.0.
        *g.state_mut(adj) = arr(&[2.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        let o = g.ref_view(out).as_slice().to_vec();
        assert!((o[0] - 2.0).abs() < 1e-12, "reinvested holdings {} != 2.0", o[0]);
    }

    /// SimpleTrader: value-weight sizing with lot rounding and a per-trade fee.
    /// Fully investing 1.0 weight of 1e6 at price 10 buys 100000 shares (1000
    /// lots = 1e6) and charges a 1000 fee (max(5, 1e6·0.001)) to cash.
    #[test]
    fn simple_trader_value_weight_with_fees_and_lots() {
        let nan = f64::NAN;
        let mut b = GraphBuilder::new();
        let pos = b.push_source(RefSource::new(arr(&[nan])));
        let close = b.push_source(RefSource::new(arr(&[nan])));
        let adj = b.push_source(RefSource::new(arr(&[1.0])));
        let up = b.push_source(RefSource::new(arr(&[nan])));
        let lo = b.push_source(RefSource::new(arr(&[nan])));
        let out = b.push(
            SimpleTrader::new(1, 1_000_000.0, 100.0, 5.0, 0.001),
            &[*pos, *close, *adj, *up, *lo][..],
        );
        let mut g = Graph::from_builder(b);
        let mut pool = Pool::new(0);

        // Tick 1: full weight fires at close 10 → one-tick delay, all cash.
        *g.state_mut(pos) = arr(&[1.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[0.0, 1_000_000.0]);

        // Tick 2: execute at close 10 → 1e6 of holdings, cash = -fee = -1000.
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool);
        assert_eq!(g.ref_view(out).as_slice(), &[1_000_000.0, -1000.0]);
    }

    /// RandomTrader: picks `portfolio_size` stocks and equal-weights them
    /// (fully invested, no fees here), is seed-deterministic, and marks to
    /// market within the held legs' bounds. With uniform weights/prices the
    /// post-rebalance book is exactly `[1000, 0]` regardless of which two are
    /// chosen; the later price move then depends on (and reveals) the seeded
    /// selection.
    #[test]
    fn random_trader_invests_and_is_seed_deterministic() {
        let nan = f64::NAN;
        let run = || {
            let mut b = GraphBuilder::new();
            let pos = b.push_source(RefSource::new(arr(&[nan; 5])));
            let close = b.push_source(RefSource::new(arr(&[nan; 5])));
            let adj = b.push_source(RefSource::new(arr(&[1.0; 5])));
            let up = b.push_source(RefSource::new(arr(&[nan; 5])));
            let lo = b.push_source(RefSource::new(arr(&[nan; 5])));
            // portfolio_size 2, lot_size 1, no fees, cash 1000, seed 0.
            let out = b.push(
                RandomTrader::new(5, 2, 1000.0, 1.0, 0.0, 0.0, 0),
                &[*pos, *close, *adj, *up, *lo][..],
            );
            let mut g = Graph::from_builder(b);
            let mut pool = Pool::new(0);
            *g.state_mut(pos) = arr(&[0.2; 5]);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool);
            let invested = g.ref_view(out).as_slice().to_vec();
            *g.state_mut(close) = arr(&[11.0, 9.0, 10.0, 10.0, 10.0]);
            g.stabilize(&mut pool);
            let marked = g.ref_view(out).as_slice().to_vec();
            (invested, marked)
        };
        let (inv1, mk1) = run();
        let (_inv2, mk2) = run();

        assert_eq!(inv1, vec![1000.0, 0.0], "should be fully invested after rebalance");
        assert_eq!(mk1, mk2, "same seed must give identical results");
        let nav = mk1[0] + mk1[1];
        assert!(nav > 900.0 && nav < 1100.0, "tick3 NAV {nav} out of held-leg bounds");
    }
}
