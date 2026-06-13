//! Stock-specific operators: `Annualize` (YTD → annualized) and `ForwardAdjust`
//! (corporate-action price adjustment, message-passing on price vs dividend
//! inputs). Implemented
//! directly on [`flowgraph::typed::Operator`] and generic over the input edge
//! kind ([`ArrayInput`]): owned `RefPort<Array<f64>>` edges (the default) or
//! zero-copy [`ArraySlice`](crate::ArraySlice) views.

use flowgraph::typed::{Interface, Operator, RefPort, ViewPort};

use super::op::{ArrayInput, ArrayValue};
use crate::Array;

// ---------------------------------------------------------------------------
// Annualize
// ---------------------------------------------------------------------------

/// Convert YTD cumulative values `[year, day_of_year, ytd_1..ytd_N]` into
/// annualized `[N]` values via days-based scaling.
pub struct Annualize<In: ArrayInput<f64> = RefPort<Array<f64>>> {
    _phantom: std::marker::PhantomData<fn() -> In>,
}

impl<In: ArrayInput<f64>> Clone for Annualize<In> {
    fn clone(&self) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

// `new` is pinned to the defaulted (owned-edge) form so the bare
// `Annualize::new()` keeps working — type-parameter defaults do not
// participate in expression inference (cf. `Vec::new` pinning `Global`).
// View-edge instantiations spell the type: [`AnnualizeView::default()`].
impl Annualize {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<In: ArrayInput<f64>> Default for Annualize<In> {
    fn default() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

/// [`Annualize`] over a zero-copy [`ArraySlice`](crate::ArraySlice) view edge
/// (e.g. a gated [`Split`](super::Split) row). Construct via `default()`.
pub type AnnualizeView = Annualize<ViewPort<ArrayValue<f64>>>;

/// Runtime state for [`Annualize`]: the previous-tick YTD snapshot plus the
/// output buffer.
pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
    out: Array<f64>,
}

impl<In: ArrayInput<f64>> Operator for Annualize<In> {
    type Inputs = In;
    type Outputs = RefPort<Array<f64>>;
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
        values: <In as Interface>::Values<'a>,
        state: &'b mut AnnualizeState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        let input = In::data(&values);
        let input = input.as_slice();
        if init {
            let input_len = input.len();
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
        _: <In as Interface>::Values<'a>,
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
/// the two inputs via the notify flags (only notified input values are read).
pub struct ForwardAdjust<P = RefPort<Array<f64>>, D = RefPort<Array<f64>>>
where
    P: ArrayInput<f64>,
    D: ArrayInput<f64>,
{
    output_prices: bool,
    _phantom: std::marker::PhantomData<fn() -> (P, D)>,
}

impl<P: ArrayInput<f64>, D: ArrayInput<f64>> Clone for ForwardAdjust<P, D> {
    fn clone(&self) -> Self {
        Self {
            output_prices: self.output_prices,
            _phantom: std::marker::PhantomData,
        }
    }
}

// `new` is pinned to the defaulted (owned-edge) form so the bare
// `ForwardAdjust::new()` keeps working (see the `Annualize::new` note);
// view-edge instantiations spell the type and use `default()`, e.g.
// [`ForwardAdjustViewDiv::default()`].
impl ForwardAdjust {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<P: ArrayInput<f64>, D: ArrayInput<f64>> ForwardAdjust<P, D> {
    pub fn with_output_prices(mut self, output_prices: bool) -> Self {
        self.output_prices = output_prices;
        self
    }
}

impl<P: ArrayInput<f64>, D: ArrayInput<f64>> Default for ForwardAdjust<P, D> {
    fn default() -> Self {
        Self {
            output_prices: true,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// [`ForwardAdjust`] whose dividend input is a zero-copy
/// [`ArraySlice`](crate::ArraySlice) view edge (e.g. a gated
/// [`Split`](super::Split) row); the price input stays an owned edge.
/// Construct via `default()`.
pub type ForwardAdjustViewDiv = ForwardAdjust<RefPort<Array<f64>>, ViewPort<ArrayValue<f64>>>;

/// Runtime state for [`ForwardAdjust`]: the adjustment factor and last price
/// plus the output buffer.
pub struct ForwardAdjustState {
    prev_price: f64,
    factor: f64,
    output_prices: bool,
    out: Array<f64>,
}

impl<P: ArrayInput<f64>, D: ArrayInput<f64>> Operator for ForwardAdjust<P, D> {
    type Inputs = (P, D);
    type Outputs = RefPort<Array<f64>>;
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
        (price_values, dividend_values): (
            <P as Interface>::Values<'a>,
            <D as Interface>::Values<'a>,
        ),
        state: &'b mut ForwardAdjustState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        let price = P::data(&price_values);
        let dividend = D::data(&dividend_values);
        if init {
            assert_eq!(price.stride(), 1, "stock price must be scalar");
            assert_eq!(
                dividend.stride(),
                2,
                "dividend data must have shape [2]: [share_dividends, cash_dividends]"
            );
            state.prev_price = f64::NAN;
            state.factor = 1.0;
            let init_val = if state.output_prices { 0.0 } else { 1.0 };
            state.out = Array::scalar(init_val);
            return (false, &state.out);
        }
        if D::notified(&dividend_values) {
            let share_dividends = dividend.as_slice()[0];
            let cash_dividends = dividend.as_slice()[1];
            let prev_price = state.prev_price;
            if !prev_price.is_nan() {
                assert!(prev_price > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_price - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        if P::notified(&price_values) {
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
        _: (
            <P as Interface>::Values<'a>,
            <D as Interface>::Values<'a>,
        ),
        state: &'b ForwardAdjustState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
    }
}
