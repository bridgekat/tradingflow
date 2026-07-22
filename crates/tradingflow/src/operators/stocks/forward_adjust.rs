use crate::data::{Array, ArrayView, Instant};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Forward price adjustment for corporate actions. Inputs: `(price, dividend)`
/// where `price` is a single value (rank `NP`, one element) and `dividend` is
/// `[share_dividends, cash_dividends]` (rank `ND`). The output mirrors the price
/// shape (rank `NP`). Message-passing on the two inputs via the notify flags
/// (only notified input values are read).
#[derive(Clone)]
pub struct ForwardAdjust<const NP: usize, const ND: usize> {
    output_prices: bool,
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

/// Runtime state for [`ForwardAdjust`]: the adjustment factor and last price
/// plus the output buffer.
pub struct ForwardAdjustState<const NP: usize> {
    prev_price: f64,
    factor: f64,
    output_prices: bool,
    out: Array<f64, NP>,
}

impl<const NP: usize, const ND: usize> Operator for ForwardAdjust<NP, ND> {
    type Inputs = (ArrayPort<f64, NP>, ArrayPort<f64, ND>);
    type Outputs = ArrayPort<f64, NP>;
    type Context = Instant;
    type State = ForwardAdjustState<NP>;

    fn init(
        self,
        ((_, price_view), (_, div_view)): (
            (bool, ArrayView<'_, f64, NP>),
            (bool, ArrayView<'_, f64, ND>),
        ),
    ) -> Self::State {
        // Only the build-time inputs' shapes are read here.
        assert_eq!(
            price_view.to_contiguous().len(),
            1,
            "stock price must be a single value"
        );
        assert_eq!(
            div_view.to_contiguous().len(),
            2,
            "dividend data must have shape [2]: [share_dividends, cash_dividends]"
        );
        let init_val = if self.output_prices { 0.0 } else { 1.0 };
        ForwardAdjustState {
            prev_price: f64::NAN,
            factor: 1.0,
            output_prices: self.output_prices,
            out: Array::from_parts(price_view.extents(), vec![init_val].into()),
        }
    }

    fn compute<'a, 'b: 'a>(
        ((price_notified, price_view), (div_notified, div_view)): (
            (bool, ArrayView<'a, f64, NP>),
            (bool, ArrayView<'a, f64, ND>),
        ),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, f64, NP>) {
        if div_notified {
            let dividend = div_view.to_contiguous();
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
            let p = price_view.to_contiguous()[0];
            state.out.data_mut()[0] = if state.output_prices {
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
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, f64, NP>) {
        (false, state.out.view())
    }
}

/// Forward price/dividend adjustment. Chain
/// [`with_output_prices`](ForwardAdjust::with_output_prices) to emit adjust
/// factors instead of adjusted prices.
pub fn forward_adjust<const NP: usize, const ND: usize>() -> ForwardAdjust<NP, ND> {
    ForwardAdjust::new()
}
