//! Stock-specific operators — port of [`crate::operators::stocks`]:
//! `Annualize` (YTD → annualized) and `ForwardAdjust` (corporate-action price
//! adjustment, message-passing on price vs dividend inputs). Implemented
//! directly on [`flowgraph::typed::Operator`].

use flowgraph::typed::{Operator, Port};

use crate::Array;

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

/// Runtime state for [`Annualize`]: the previous-tick YTD snapshot plus the
/// output buffer.
pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
    out: Array<f64>,
}

impl Operator for Annualize {
    type Inputs = Port<Array<f64>>;
    type Outputs = Port<Array<f64>>;
    type State = AnnualizeState;

    fn init(self) -> AnnualizeState {
        AnnualizeState {
            prev_ytd: Vec::new(),
            prev_year: 0,
            prev_day: 0.0,
            initialized: false,
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, inputs): (bool, &'a Array<f64>),
        state: &'b mut AnnualizeState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            let input_len = inputs.as_slice().len();
            assert!(
                input_len >= 3,
                "Annualize: input must have shape [2 + N] with N >= 1, got length {input_len}"
            );
            let n = input_len - 2;
            state.prev_ytd = vec![0.0; n];
            state.prev_year = 0;
            state.prev_day = 0.0;
            state.initialized = false;
            state.out = Array::zeros(&[n]);
            return (false, &state.out);
        }
        let input = inputs.as_slice();
        let year = input[0].floor() as i64;
        let day = input[1];
        let ytd = &input[2..];
        let n = ytd.len();
        let out = state.out.as_mut_slice();

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
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<f64>),
        state: &'b AnnualizeState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// ForwardAdjust
// ---------------------------------------------------------------------------

/// Forward price adjustment for corporate actions. Inputs: `(price, dividend)`
/// where dividend is `[share_dividends, cash_dividends]`. Message-passing on
/// the two inputs via the notify flags.
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

/// Runtime state for [`ForwardAdjust`]: the adjustment factor and last price
/// plus the output buffer.
pub struct ForwardAdjustState {
    prev_price: f64,
    factor: f64,
    output_prices: bool,
    out: Array<f64>,
}

impl Operator for ForwardAdjust {
    type Inputs = (Port<Array<f64>>, Port<Array<f64>>);
    type Outputs = Port<Array<f64>>;
    type State = ForwardAdjustState;

    fn init(self) -> ForwardAdjustState {
        ForwardAdjustState {
            prev_price: f64::NAN,
            factor: 1.0,
            output_prices: self.output_prices,
            out: Array::scalar(0.0),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((produced_price, price), (produced_dividend, dividend)): (
            (bool, &'a Array<f64>),
            (bool, &'a Array<f64>),
        ),
        state: &'b mut ForwardAdjustState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            assert_eq!(price.as_slice().len(), 1, "stock price must be scalar");
            assert_eq!(
                dividend.as_slice().len(),
                2,
                "dividend data must have shape [2]: [share_dividends, cash_dividends]"
            );
            state.prev_price = f64::NAN;
            state.factor = 1.0;
            let init_val = if state.output_prices { 0.0 } else { 1.0 };
            state.out = Array::scalar(init_val);
            return (false, &state.out);
        }
        if produced_dividend {
            let share_dividends = dividend.as_slice()[0];
            let cash_dividends = dividend.as_slice()[1];
            let prev_price = state.prev_price;
            if !prev_price.is_nan() {
                assert!(prev_price > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_price - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        if produced_price {
            let price = price.as_slice()[0];
            state.out.as_mut_slice()[0] = if state.output_prices {
                price * state.factor
            } else {
                state.factor
            };
            state.prev_price = price;
            (true, &state.out)
        } else {
            (false, &state.out)
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: ((bool, &'a Array<f64>), (bool, &'a Array<f64>)),
        state: &'b ForwardAdjustState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
    }
}
