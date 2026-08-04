use num_traits::Float;
use std::cmp::Ordering;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

pub struct Scale<T: Scalar + Float, const N: usize> {
    compare: Ordering,
    target: f64,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Scale<T, N> {
    pub fn new(compare: Ordering, target: f64) -> Self {
        Self {
            compare,
            target,
            _marker: PhantomData,
        }
    }
}

pub struct ScaleState<T: Scalar + Float, const N: usize> {
    compare: Ordering,
    target: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Scale<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ScaleState<T, N>;

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let mut state = ScaleState {
            compare: self.compare,
            target: T::from(self.target).unwrap(),
            out: Array::zeros(x.extents()),
        };
        scale_into(&mut state.out, x, state.compare, state.target);
        state
    }

    fn reset<'a, 'b: 'a>(
        _: ArrayView<'a, T, N>,
        state: &'b mut Self::State,
    ) -> ArrayView<'a, T, N> {
        state.out.view()
    }

    fn compute<'a, 'b: 'a>(
        x: ArrayView<'a, T, N>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, T, N> {
        scale_into(&mut state.out, x, state.compare, state.target);
        state.out.view()
    }
}

fn scale_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, N>,
    x: ArrayView<'_, T, N>,
    compare: Ordering,
    target: T,
) {
    let mut sum = T::zero();
    for &v in x.iter() {
        if v.is_finite() {
            sum = sum + v.abs();
        }
    }
    let factor = if sum.partial_cmp(&target) == Some(compare) {
        T::one() // Skip scaling if the sum meets the condition.
    } else if sum > T::zero() {
        target / sum // Scale to reach the target sum.
    } else {
        T::nan() // A zero (or empty) cross-section cannot reach the target.
    };
    let o = out.data_mut();
    for (i, &v) in x.iter().enumerate() {
        o[i] = if v.is_finite() { v * factor } else { T::nan() };
    }
}

/// Cross-sectional rescale: `x · target / Σ|x|`, so the absolute values of
/// the output sum to `target` (Alpha101's `scale(x, a)`). Non-finite entries
/// (NaN or ±∞) are treated as missing: they map to NaN and are excluded from
/// the sum. When the absolute sum is zero the whole cross-section is NaN.
pub fn scale<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Equal, target)
}

/// Like [`scale`], but only scale when the sum is no greater than `target`.
pub fn scale_up<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Greater, target)
}

/// Like [`scale`], but only scale when the sum is no less than `target`.
pub fn scale_down<T: Scalar + Float, const N: usize>(
    target: f64,
) -> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Scale::new(Ordering::Less, target)
}
