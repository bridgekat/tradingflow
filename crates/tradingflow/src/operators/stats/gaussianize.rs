use num_traits::Float;
use statrs::distribution::{ContinuousCDF, Normal};
use std::marker::PhantomData;

use super::rank::rank_positions;
use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::Operator;
use crate::ports::ArrayPort;

pub struct Gaussianize<T: Scalar + Float, const N: usize> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: Scalar + Float, const N: usize> Gaussianize<T, N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Gaussianize<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GaussianizeState<T: Scalar + Float, const N: usize> {
    idx: Vec<usize>,
    pos: Vec<f64>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Gaussianize<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GaussianizeState<T, N>;

    fn init(self, x: ArrayView<'_, T, N>) -> Self::State {
        let len = x.layout().len();
        let mut state = GaussianizeState {
            idx: vec![0; len],
            pos: vec![0.0; len],
            out: Array::zeros(x.extents()),
        };
        gaussianize_into(&mut state, x);
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
        gaussianize_into(state, x);
        state.out.view()
    }
}

fn gaussianize_into<T: Scalar + Float, const N: usize>(
    state: &mut GaussianizeState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;

    let GaussianizeState { idx, pos, out } = state;
    let n_valid = rank_positions(src, idx, pos) as f64;

    // Standard normal; `new` only fails for a non-positive scale.
    let normal = Normal::new(0.0, 1.0).expect("standard normal is well-defined");
    let nan = T::nan();
    for (slot, &p) in out.data_mut().iter_mut().zip(pos.iter()) {
        *slot = if p.is_nan() {
            nan
        } else {
            let z = normal.inverse_cdf((p + 0.5) / n_valid);
            T::from(z).unwrap_or(nan)
        };
    }
}

/// Cross-sectional rank mapped through the inverse normal CDF. Non-finite
/// entries (NaN or ±∞) are treated as missing and map to NaN; tied values
/// share their average rank.
pub fn gaussianize<T: Scalar + Float, const N: usize>()
-> impl Operator<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Gaussianize::new()
}
