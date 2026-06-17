//! Stock-specific operators: `Annualize` (YTD → annualized) and `ForwardAdjust`
//! (corporate-action price adjustment, message-passing on price vs dividend
//! inputs), over the strided [`ArrayView`] currency.

use flowgraph::typed::{Operator, ViewPort};

use super::op::ArrayValue;
use crate::{Array, ArrayView};

// ---------------------------------------------------------------------------
// Annualize
// ---------------------------------------------------------------------------

/// Convert a YTD cumulative vector `[year, day_of_year, ytd_1..ytd_N]` into
/// annualized `[N]` values via days-based scaling (rank-1 in / rank-1 out).
#[derive(Clone, Default)]
pub struct Annualize;

impl Annualize {
    pub fn new() -> Self {
        Self
    }
}

/// [`Annualize`] over a view edge — now the same operator (the currency is
/// already views); retained for source compatibility. Construct via `default()`.
pub type AnnualizeView = Annualize;

/// Runtime state for [`Annualize`]: the previous-tick YTD snapshot plus the
/// output buffer.
pub struct AnnualizeState {
    prev_ytd: Vec<f64>,
    prev_year: i64,
    prev_day: f64,
    initialized: bool,
    out: Array<f64, 1>,
}

impl Operator for Annualize {
    type Inputs = ViewPort<ArrayValue<f64, 1>>;
    type Outputs = ViewPort<ArrayValue<f64, 1>>;
    type State = AnnualizeState;

    fn init(self) -> AnnualizeState {
        AnnualizeState {
            prev_ytd: Vec::new(),
            prev_year: 0,
            prev_day: 0.0,
            initialized: false,
            out: Array::zeros([0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, view): (bool, ArrayView<'a, f64, 1>),
        state: &'b mut AnnualizeState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        let xs = view.to_contiguous();
        let input: &[f64] = &xs;
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
            state.out = Array::zeros([n]);
            return (false, state.out.view());
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
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, 1>),
        state: &'b AnnualizeState,
    ) -> (bool, ArrayView<'a, f64, 1>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// ForwardAdjust
// ---------------------------------------------------------------------------

/// Forward price adjustment for corporate actions. Inputs: `(price, dividend)`
/// where `price` is a single value (rank `NP`, one element) and `dividend` is
/// `[share_dividends, cash_dividends]` (rank `ND`). The output mirrors the price
/// shape (rank `NP`). Message-passing on the two inputs via the notify flags
/// (only notified input values are read).
pub struct ForwardAdjust<const NP: usize, const ND: usize> {
    output_prices: bool,
}

impl<const NP: usize, const ND: usize> Clone for ForwardAdjust<NP, ND> {
    fn clone(&self) -> Self {
        Self {
            output_prices: self.output_prices,
        }
    }
}

impl<const NP: usize, const ND: usize> ForwardAdjust<NP, ND> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_output_prices(mut self, output_prices: bool) -> Self {
        self.output_prices = output_prices;
        self
    }
}

impl<const NP: usize, const ND: usize> Default for ForwardAdjust<NP, ND> {
    fn default() -> Self {
        Self {
            output_prices: true,
        }
    }
}

/// [`ForwardAdjust`] whose dividend input is a view edge — now the same operator
/// (the currency is already views). Construct via `default()`.
pub type ForwardAdjustViewDiv<const NP: usize, const ND: usize> = ForwardAdjust<NP, ND>;

/// Runtime state for [`ForwardAdjust`]: the adjustment factor and last price
/// plus the output buffer.
pub struct ForwardAdjustState<const NP: usize> {
    prev_price: f64,
    factor: f64,
    output_prices: bool,
    out: Array<f64, NP>,
}

impl<const NP: usize, const ND: usize> Operator for ForwardAdjust<NP, ND> {
    type Inputs = (ViewPort<ArrayValue<f64, NP>>, ViewPort<ArrayValue<f64, ND>>);
    type Outputs = ViewPort<ArrayValue<f64, NP>>;
    type State = ForwardAdjustState<NP>;

    fn init(self) -> Self::State {
        ForwardAdjustState {
            prev_price: f64::NAN,
            factor: 1.0,
            output_prices: self.output_prices,
            out: Array::zeros([0; NP]),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((price_notified, price_view), (div_notified, div_view)): (
            (bool, ArrayView<'a, f64, NP>),
            (bool, ArrayView<'a, f64, ND>),
        ),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, NP>) {
        let price = price_view.to_contiguous();
        let dividend = div_view.to_contiguous();
        if init {
            assert_eq!(price.len(), 1, "stock price must be a single value");
            assert_eq!(
                dividend.len(),
                2,
                "dividend data must have shape [2]: [share_dividends, cash_dividends]"
            );
            state.prev_price = f64::NAN;
            state.factor = 1.0;
            let init_val = if state.output_prices { 0.0 } else { 1.0 };
            state.out = Array::from_vec(price_view.extents(), vec![init_val]);
            return (false, state.out.view());
        }
        if div_notified {
            let share_dividends = dividend[0];
            let cash_dividends = dividend[1];
            let prev_price = state.prev_price;
            if !prev_price.is_nan() {
                assert!(prev_price > cash_dividends);
                state.factor *= 1.0 + cash_dividends / (prev_price - cash_dividends);
                state.factor *= 1.0 + share_dividends;
            }
        }
        if price_notified {
            let p = price[0];
            state.out.as_mut_slice()[0] = if state.output_prices {
                p * state.factor
            } else {
                state.factor
            };
            state.prev_price = p;
            (true, state.out.view())
        } else {
            (false, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (
            (bool, ArrayView<'a, f64, NP>),
            (bool, ArrayView<'a, f64, ND>),
        ),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, f64, NP>) {
        (false, state.out.view())
    }
}
