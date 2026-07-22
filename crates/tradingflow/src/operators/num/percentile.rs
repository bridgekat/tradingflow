//! [`Percentile`] — the cross-sectional rank → percentile transform.

use std::marker::PhantomData;

use num_traits::Float;

use super::rank::rank_finite;
use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Cross-sectional rank-to-percentile transform over the flat cross-section.
#[derive(Clone)]
pub struct Percentile<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> Percentile<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for Percentile<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Percentile`]: the index sort scratch buffer plus the
/// output buffer.
pub struct PercentileState<T: Scalar + Float, const N: usize> {
    scratch: Vec<usize>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Percentile<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = PercentileState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = PercentileState {
            scratch: vec![0; x.layout().len()],
            out: Array::zeros(x.extents()),
        };
        // Seed the output with the faithful transform of the build value; the
        // initial render does not notify.
        percentile_into(&mut state, x);
        state
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        percentile_into(state, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Percentile`]: rank the finite entries of `x` into
/// `state.out` as percentiles in `[0, 1]` (NaN elsewhere).
fn percentile_into<T: Scalar + Float, const N: usize>(
    state: &mut PercentileState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let xs = x.to_contiguous();
    let src: &[T] = &xs;

    let PercentileState { scratch, out } = state;
    let dst = out.data_mut();
    let n_valid = rank_finite(src, scratch, dst);

    if n_valid > 0 {
        let nan = T::nan();
        let denom = T::from(n_valid as f64).unwrap_or(nan);
        let half = T::from(0.5).unwrap_or(nan);
        for rank in 0..n_valid {
            let p = (T::from(rank as f64).unwrap_or(nan) + half) / denom;
            dst[scratch[rank]] = p;
        }
    }
}

/// Cross-sectional percentile rank into `[0, 1]`.
pub fn percentile<T: Scalar + Float, const N: usize>() -> Percentile<T, N> {
    Percentile::new()
}
