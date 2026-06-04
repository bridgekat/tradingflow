//! Stock-specific operators — port of [`crate::operators::stocks`]:
//! `Annualize` (YTD → annualized) and `ForwardAdjust` (corporate-action price
//! adjustment, message-passing on price vs dividend inputs).

use flowgraph::typed::Port;

use super::op::Operator;
use crate::{Array, Instant};

// ---------------------------------------------------------------------------
// Annualize
// ---------------------------------------------------------------------------

/// Convert YTD cumulative values `[year, day_of_year, ytd_1..ytd_N]` into
/// annualized `[N]` values via days-based scaling.
#[derive(Clone)]
pub struct Annualize;

impl Annualize {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Annualize {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
}

impl Operator for Annualize {
    type State = AnnualizeState;
    type Inputs = Port<Array<f64>>;
    type Output = Array<f64>;

    fn init(&self, inputs: &Array<f64>, _ts: Instant) -> (AnnualizeState, Array<f64>) {
        let input_len = inputs.as_slice().len();
        assert!(
            input_len >= 3,
            "Annualize: input must have shape [2 + N] with N >= 1, got length {input_len}"
        );
        let n = input_len - 2;
        let state = AnnualizeState {
            prev_ytd: vec![0.0; n],
            prev_year: 0,
            prev_day: 0.0,
            initialized: false,
        };
        (state, Array::zeros(&[n]))
    }

    fn compute(
        state: &mut AnnualizeState,
        inputs: &Array<f64>,
        output: &mut Array<f64>,
        _ts: Instant,
        _produced: bool,
    ) -> bool {
        let input = inputs.as_slice();
        let year = input[0].floor() as i64;
        let day = input[1];
        let ytd = &input[2..];
        let n = ytd.len();
        let out = output.as_mut_slice();

        let (is_new_year, days_elapsed) = if !state.initialized || year != state.prev_year {
            (true, day)
        } else {
            (false, day - state.prev_day)
        };

        if days_elapsed <= 0.0 {
            for o in out.iter_mut() {
                *o = f64::NAN;
            }
        } else {
            let scale = 365.0 / days_elapsed;
            for i in 0..n {
                let delta = if is_new_year {
                    ytd[i]
                } else {
                    ytd[i] - state.prev_ytd[i]
                };
                out[i] = delta * scale;
            }
        }

        state.prev_ytd.copy_from_slice(ytd);
        state.prev_year = year;
        state.prev_day = day;
        state.initialized = true;
        true
    }
}

// ---------------------------------------------------------------------------
// ForwardAdjust
// ---------------------------------------------------------------------------

/// Forward price adjustment for corporate actions. Inputs: `(price, dividend)`
/// where dividend is `[share_dividends, cash_dividends]`. Message-passing on
/// the two inputs via the notify tree.
#[derive(Clone)]
pub struct ForwardAdjust {
    output_prices: bool,
}

impl ForwardAdjust {
    pub fn new() -> Self {
        Self {
            output_prices: true,
        }
    }

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
    type Inputs = (Port<Array<f64>>, Port<Array<f64>>);
    type Output = Array<f64>;

    fn init(
        &self,
        inputs: (&Array<f64>, &Array<f64>),
        _ts: Instant,
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
        _ts: Instant,
        produced: (bool, bool),
    ) -> bool {
        let (produced_price, produced_dividend) = produced;
        if produced_dividend {
            let share_dividends = inputs.1.as_slice()[0];
            let cash_dividends = inputs.1.as_slice()[1];
            let prev_price = state.prev_price;
            if !prev_price.is_nan() {
                assert!(prev_price > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_price - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        if produced_price {
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
