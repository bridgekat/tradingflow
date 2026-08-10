use num_traits::Float;
use std::marker::PhantomData;

use super::common::is_constant;
use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

/// Operator signature for [`standardize`] etc.
pub struct Standardize<T: Scalar + Float, const N: usize> {
    zero_degenerate: bool,
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Standardize<T, N> {
    pub fn new(zero_degenerate: bool) -> Self {
        Self {
            zero_degenerate,
            _marker: PhantomData,
        }
    }
}

/// Runtime state for [`standardize`].
pub struct StandardizeState<T: Scalar + Float, const N: usize> {
    zero_degenerate: bool,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Standardize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = StandardizeState<T, N>;

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let mut state = StandardizeState {
            zero_degenerate: self.zero_degenerate,
            out: Array::full(x.extents(), T::nan()),
        };
        standardize_into(&mut state.out, x, state.zero_degenerate);
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
        standardize_into(&mut state.out, x, state.zero_degenerate);
        state.out.view()
    }
}

fn standardize_into<T: Scalar + Float, const N: usize>(
    out: &mut Array<T, N>,
    xs: ArrayView<'_, T, N>,
    zero_degenerate: bool,
) {
    let mut count = T::zero();
    let mut sum = T::zero();
    for &x in xs.iter() {
        if x.is_finite() {
            count = count + T::one();
            sum = sum + x;
        }
    }
    let mean = sum / count;
    let mut ssd = T::zero();
    for &x in xs.iter() {
        if x.is_finite() {
            let dx = x - mean;
            ssd = ssd + dx * dx;
        }
    }
    let var = ssd / (count - T::one());
    if count < T::from(2).unwrap() || is_constant(var, mean, count) {
        for (o, &x) in out.data_mut().iter_mut().zip(xs.iter()) {
            *o = if x.is_finite() && zero_degenerate {
                T::zero()
            } else {
                T::nan()
            };
        }
    } else {
        let std = var.sqrt();
        for (o, &x) in out.data_mut().iter_mut().zip(xs.iter()) {
            *o = if x.is_finite() {
                (x - mean) / std
            } else {
                T::nan()
            };
        }
    }
}

/// Cross-sectional z-score `(x − mean) / std`, over the finite entries, with
/// the sample standard deviation.
///
/// Non-finite entries map to `NaN`, and the whole cross-section is `NaN` if
/// fewer than two finite entries remain or they have close to no spread.
pub fn standardize<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Standardize::new(false)
}

/// Like [`standardize`], but a degenerate cross-section — too few finite
/// entries, or no spread beyond floating-point error — maps its **finite**
/// entries to `0`, the neutral z-score, instead of `NaN`; non-finite entries
/// still map to `NaN`, so missing stays missing.
///
/// This is the preprocessing form (scikit-learn's `StandardScaler` makes the
/// same choice): a dead column becomes exactly zero over the entries a model
/// fits on — inert in the normal equations, zeroed by a ridge, truncated
/// cleanly by a pseudo-inverse — rather than `NaN` that a downstream
/// `isfinite` mask would let poison every row it touches. Keep plain
/// [`standardize`] for statistics and diagnostics, where "undefined" is the
/// honest answer and `NaN` propagation is load-bearing.
pub fn standardize_or_zero<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Standardize::new(true)
}
