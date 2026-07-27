//! The frictionless [`Benchmark`] executor.

use super::core::{TraderInputs, TraderValues, Vp};
use crate::data::{Array, ArrayView, Instant, Layout};
use crate::graph::Segment;
use crate::ports::is_eventful;

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

impl Segment for Benchmark {
    type Inputs = TraderInputs;
    type Outputs = Vp;
    type Context = Instant;
    type State = BenchmarkState;

    fn init(self, (pos, ..): TraderValues<'_>) -> BenchmarkState {
        let n = self.num_stocks;
        assert_eq!(
            pos.layout().len(),
            n,
            "Benchmark: input length {} != num_stocks {n}",
            pos.layout().len(),
        );
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
        (pos, close, adj, up, lo): TraderValues<'a>,
        state: &'b mut BenchmarkState,
        _: &Instant,
    ) -> ArrayView<'a, f64, 1> {
        let n = state.num_stocks;
        let pos_eventful = is_eventful(pos);
        let can_exec = is_eventful(close);
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
        // Only an eventful close batch can execute: with no close events there
        // is no market to trade in, and the pending signal is held.
        if can_exec && let Some(pending) = state.pending.take() {
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

        // Capture a new target for execution on the NEXT eventful close.
        if pos_eventful {
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
        state.out.view()
    }

    fn reset<'a, 'b: 'a>(
        _: TraderValues<'a>,
        state: &'b mut BenchmarkState,
    ) -> ArrayView<'a, f64, 1> {
        state.out.view()
    }
}

/// Frictionless benchmark executor: replicates target weights exactly.
pub fn benchmark(num_stocks: usize, initial_cash: f64, use_adjusts: bool) -> Benchmark {
    Benchmark::new(num_stocks, initial_cash, use_adjusts)
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{arr, src};
    use super::*;
    use crate::graph::pool::Pool;
    use crate::graph::typed::Builder;

    #[test]
    fn benchmark_one_tick_delay_exec_and_mark() {
        let nan = f64::NAN;
        let mut b = Builder::new();
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
        g.stabilize(&mut pool, &Instant::MIN);
        assert_eq!(g.view(out).as_slice().unwrap(), &[0.0, 1.0]);

        *g.state_mut(close) = arr(&[11.0, 22.0]);
        g.stabilize(&mut pool, &Instant::MIN);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!((o[0] - 1.0).abs() < 1e-12, "tick2 holdings {} != 1.0", o[0]);
        assert!(o[1].abs() < 1e-12, "tick2 cash {} != 0", o[1]);

        *g.state_mut(close) = arr(&[12.0, 22.0]);
        g.stabilize(&mut pool, &Instant::MIN);
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
        let mut b = Builder::new();
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
        g.stabilize(&mut pool, &Instant::MIN);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool, &Instant::MIN);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!(
            (o[0] - 1.0).abs() < 1e-12 && o[1].abs() < 1e-12,
            "fully invested: {o:?}"
        );

        *g.state_mut(adj) = arr(&[2.0]);
        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool, &Instant::MIN);
        let o = g.view(out).as_slice().unwrap().to_vec();
        assert!(
            (o[0] - 2.0).abs() < 1e-12,
            "reinvested holdings {} != 2.0",
            o[0]
        );
    }
}
