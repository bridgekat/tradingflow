//! The realistic-cost, value-weighted [`SimpleTrader`] executor.

use crate::data::{ArrayView, Instant, Layout};
use crate::graph::typed::Operator;

use super::core::{TraderCore, TraderInputs, TraderValues, Vp, run_trader};

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

    fn init(self, ((_, pos), ..): TraderValues<'_>) -> SimpleTraderState {
        assert_eq!(
            pos.layout().len(),
            self.num_stocks,
            "trader: input length {} != num_stocks {}",
            pos.layout().len(),
            self.num_stocks,
        );
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
        state: &'b mut SimpleTraderState,
        _: &Instant,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        run_trader(&mut state.core, values, value_weight_lots)
    }

    fn passthrough<'a, 'b: 'a>(
        _: TraderValues<'a>,
        state: &'b mut SimpleTraderState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.core.out.view())
    }
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

#[cfg(test)]
mod tests {
    use super::super::test_util::{arr, src};
    use super::*;
    use crate::graph::pool::Pool;
    use crate::graph::typed::Builder;

    #[test]
    fn simple_trader_value_weight_with_fees_and_lots() {
        let nan = f64::NAN;
        let mut b = Builder::new();
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
        g.stabilize(&mut pool, &Instant::MIN);
        assert_eq!(g.view(out).as_slice().unwrap(), &[0.0, 1_000_000.0]);

        *g.state_mut(close) = arr(&[10.0]);
        g.stabilize(&mut pool, &Instant::MIN);
        assert_eq!(g.view(out).as_slice().unwrap(), &[1_000_000.0, -1000.0]);
    }
}
