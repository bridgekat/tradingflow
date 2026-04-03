//! Forward price adjustment operator for corporate actions.
//!
//! [`ForwardAdjust`] accumulates a multiplicative adjustment factor from
//! share dividends (bonus shares) and cash dividends, producing a
//! forward-adjusted close price on every tick.  Uses [`Notify`] to detect
//! when dividend data arrives.

use crate::operator::Notify;
use crate::{Array, Operator};

/// Compute the forward-adjusted close price from a raw close price and
/// dividend data for a single stock.
///
/// # Inputs
///
/// 0. Close price: scalar `Array<f64>` (shape `[]`).
/// 1. Dividend data: `Array<f64>` of shape `[2]`, laid out as
///    `[share_dividends, cash_dividends]`.
///
/// When the dividend input updates ([`Notify::input_produced`]), the operator
/// computes an adjustment multiplier using the *previous* close price
/// (stored in state) and accumulates it into the cumulative factor.
///
/// # Output
///
/// Forward-adjusted close price: scalar `Array<f64>` (shape `[]`), equal
/// to `raw_close * cumulative_factor`.
pub struct ForwardAdjust;

impl ForwardAdjust {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ForwardAdjust {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ForwardAdjustState {
    prev_close: f64,
    factor: f64,
}

impl Operator for ForwardAdjust {
    type State = ForwardAdjustState;
    type Inputs = (Array<f64>, Array<f64>);
    type Output = Array<f64>;

    fn init(
        self,
        inputs: (&Array<f64>, &Array<f64>),
        _timestamp: i64,
    ) -> (ForwardAdjustState, Array<f64>) {
        assert_eq!(inputs.0.as_slice().len(), 1, "close price must be scalar");
        assert_eq!(
            inputs.1.as_slice().len(),
            2,
            "dividend data must have shape [2]: [share_dividends, cash_dividends]"
        );
        let state = ForwardAdjustState {
            prev_close: f64::NAN,
            factor: 1.0,
        };
        (state, Array::scalar(0.0))
    }

    fn compute(
        state: &mut ForwardAdjustState,
        inputs: (&Array<f64>, &Array<f64>),
        output: &mut Array<f64>,
        _timestamp: i64,
        notify: &Notify<'_>,
    ) -> bool {
        if notify.input_produced(1) {
            let share_dividends = inputs.1.as_slice()[0];
            let cash_dividends = inputs.1.as_slice()[1];
            let prev_close = state.prev_close;
            if !prev_close.is_nan() {
                assert!(prev_close > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_close - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        let close = inputs.0.as_slice()[0];
        state.prev_close = close;
        output.as_mut_slice()[0] = close * state.factor;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn price(v: f64) -> Array<f64> {
        Array::scalar(v)
    }
    fn div(sd: f64, cd: f64) -> Array<f64> {
        Array::from_vec(&[2], vec![sd, cd])
    }
    fn no_div() -> Array<f64> {
        div(0.0, 0.0)
    }
    fn notify_price() -> Notify<'static> {
        Notify::new(&[true, false], &[0, 1])
    }
    fn notify_both() -> Notify<'static> {
        Notify::new(&[true, true], &[0, 1])
    }

    #[test]
    fn no_dividends() {
        let p = price(10.0);
        let d = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p, &d), 0);
        ForwardAdjust::compute(&mut s, (&p, &d), &mut o, 1, &notify_price());
        assert_eq!(o.as_slice(), &[10.0]);
    }

    #[test]
    fn cash_dividend() {
        let p0 = price(10.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), 0);

        // Day 1: establish previous close.
        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, 1, &notify_price());
        assert_eq!(o.as_slice(), &[10.0]);

        // Day 2 (ex-date): price drops to 9.5, cash dividend = 0.5.
        // Multiplier = (1 + 0.5 / (10.0 - 0.5)) * 1 = 1 + 0.5/9.5.
        let p1 = price(9.5);
        let d1 = div(0.0, 0.5);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, 2, &notify_both());

        let factor = 1.0 + 0.5 / 9.5;
        assert!((o.as_slice()[0] - 9.5 * factor).abs() < 1e-12);
        // Adjusted price ≈ previous unadjusted close (10.0).
        assert!((o.as_slice()[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn share_dividend() {
        let p0 = price(20.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), 0);

        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, 1, &notify_price());

        // 10% bonus shares: factor = 1.1.
        let p1 = price(18.18);
        let d1 = div(0.1, 0.0);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, 2, &notify_both());

        assert!((o.as_slice()[0] - 18.18 * 1.1).abs() < 1e-12);
    }

    #[test]
    fn cumulative_dividends() {
        let p0 = price(100.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), 0);

        // Day 1.
        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, 1, &notify_price());

        // Day 2: cash dividend = 2.0 on close = 100.
        let p1 = price(98.0);
        let d1 = div(0.0, 2.0);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, 2, &notify_both());
        let f1 = 1.0 + 2.0 / (100.0 - 2.0);

        // Day 3: normal trading.
        let p2 = price(99.0);
        ForwardAdjust::compute(&mut s, (&p2, &d0), &mut o, 3, &notify_price());
        assert!((o.as_slice()[0] - 99.0 * f1).abs() < 1e-12);

        // Day 4: another cash dividend = 1.0 on close = 99.
        let p3 = price(98.0);
        let d3 = div(0.0, 1.0);
        ForwardAdjust::compute(&mut s, (&p3, &d3), &mut o, 4, &notify_both());
        let f2 = f1 * (1.0 + 1.0 / (99.0 - 1.0));
        assert!((o.as_slice()[0] - 98.0 * f2).abs() < 1e-12);
    }
}
