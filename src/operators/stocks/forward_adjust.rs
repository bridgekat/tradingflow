//! Forward price adjustment operator for corporate actions.
//!
//! [`ForwardAdjust`] accumulates a multiplicative adjustment factor from
//! share dividends (bonus shares) and cash dividends, producing either a
//! forward-adjusted stock price or the raw adjustment factor on every tick.

use crate::time::Instant;
use crate::{Array, Notify, Operator};

/// Compute the forward-adjusted stock price (or adjustment factor) from
/// a raw stock price and dividend data for a single stock.
///
/// # Inputs
///
/// 0. Stock price: scalar `Array<f64>` (shape `[]`).
/// 1. Dividend data: `Array<f64>` of shape `[2]`, laid out as
///    `[share_dividends, cash_dividends]`.
///
/// When the dividend input updates ([`Notify::input_produced`]), the operator
/// computes an adjustment multiplier using the *previous* stock price
/// (stored in state) and accumulates it into the cumulative factor.
///
/// # Output
///
/// If `output_prices` is `true` (default): forward-adjusted stock price
/// (`raw_price * factor`).  If `false`: the cumulative adjustment factor
/// itself.
pub struct ForwardAdjust {
    output_prices: bool,
}

impl ForwardAdjust {
    pub fn new() -> Self {
        Self {
            output_prices: true,
        }
    }

    /// If `false`, output the cumulative adjustment factor instead of
    /// the adjusted price.
    pub fn with_output_prices(mut self, output_prices: bool) -> Self {
        self.output_prices = output_prices;
        self
    }
}

impl Default for ForwardAdjust {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ForwardAdjustState {
    prev_price: f64,
    factor: f64,
    output_prices: bool,
}

impl Operator for ForwardAdjust {
    type State = ForwardAdjustState;
    type Inputs = (Array<f64>, Array<f64>);
    type Output = Array<f64>;

    fn init(
        self,
        inputs: (&Array<f64>, &Array<f64>),
        _timestamp: Instant,
    ) -> (ForwardAdjustState, Array<f64>) {
        assert_eq!(inputs.0.as_slice().len(), 1, "stock price must be scalar");
        assert_eq!(
            inputs.1.as_slice().len(),
            2,
            "dividend data must have shape [2]: [share_dividends, cash_dividends]"
        );
        let state = ForwardAdjustState {
            prev_price: f64::NAN,
            factor: 1.0,
            output_prices: self.output_prices,
        };
        let init_val = if self.output_prices { 0.0 } else { 1.0 };
        (state, Array::scalar(init_val))
    }

    fn compute(
        state: &mut ForwardAdjustState,
        inputs: (&Array<f64>, &Array<f64>),
        output: &mut Array<f64>,
        _timestamp: Instant,
        notify: &Notify<'_>,
    ) -> bool {
        let input_produced = notify.input_produced();
        if input_produced[1] {
            let share_dividends = inputs.1.as_slice()[0];
            let cash_dividends = inputs.1.as_slice()[1];
            let prev_price = state.prev_price;
            if !prev_price.is_nan() {
                assert!(prev_price > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_price - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        if input_produced[0] {
            let price = inputs.0.as_slice()[0];
            output.as_mut_slice()[0] = if state.output_prices {
                price * state.factor
            } else {
                state.factor
            };
            state.prev_price = price;
            true
        } else {
            false
        }
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
        Notify::new(&[0], 2)
    }
    fn notify_both() -> Notify<'static> {
        Notify::new(&[0, 1], 2)
    }

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn no_dividends() {
        let p = price(10.0);
        let d = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p, &d), ts(0));
        ForwardAdjust::compute(&mut s, (&p, &d), &mut o, ts(1), &notify_price());
        assert_eq!(o.as_slice(), &[10.0]);
    }

    #[test]
    fn cash_dividend() {
        let p0 = price(10.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), ts(0));

        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, ts(1), &notify_price());
        assert_eq!(o.as_slice(), &[10.0]);

        let p1 = price(9.5);
        let d1 = div(0.0, 0.5);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, ts(2), &notify_both());

        let factor = 1.0 + 0.5 / 9.5;
        assert!((o.as_slice()[0] - 9.5 * factor).abs() < 1e-12);
        assert!((o.as_slice()[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn share_dividend() {
        let p0 = price(20.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), ts(0));

        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, ts(1), &notify_price());

        let p1 = price(18.18);
        let d1 = div(0.1, 0.0);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, ts(2), &notify_both());

        assert!((o.as_slice()[0] - 18.18 * 1.1).abs() < 1e-12);
    }

    #[test]
    fn cumulative_dividends() {
        let p0 = price(100.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new().init((&p0, &d0), ts(0));

        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, ts(1), &notify_price());

        let p1 = price(98.0);
        let d1 = div(0.0, 2.0);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, ts(2), &notify_both());
        let f1 = 1.0 + 2.0 / (100.0 - 2.0);

        let p2 = price(99.0);
        ForwardAdjust::compute(&mut s, (&p2, &d0), &mut o, ts(3), &notify_price());
        assert!((o.as_slice()[0] - 99.0 * f1).abs() < 1e-12);

        let p3 = price(98.0);
        let d3 = div(0.0, 1.0);
        ForwardAdjust::compute(&mut s, (&p3, &d3), &mut o, ts(4), &notify_both());
        let f2 = f1 * (1.0 + 1.0 / (99.0 - 1.0));
        assert!((o.as_slice()[0] - 98.0 * f2).abs() < 1e-12);
    }

    #[test]
    fn output_factor() {
        let p0 = price(100.0);
        let d0 = no_div();
        let (mut s, mut o) = ForwardAdjust::new()
            .with_output_prices(false)
            .init((&p0, &d0), ts(0));

        ForwardAdjust::compute(&mut s, (&p0, &d0), &mut o, ts(1), &notify_price());
        assert_eq!(o.as_slice()[0], 1.0);

        let p1 = price(98.0);
        let d1 = div(0.0, 2.0);
        ForwardAdjust::compute(&mut s, (&p1, &d1), &mut o, ts(2), &notify_both());
        let expected = 1.0 + 2.0 / (100.0 - 2.0);
        assert!((o.as_slice()[0] - expected).abs() < 1e-12);
    }
}
