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
}
