//! [`Clamp`] — element-wise clipping to a constant interval.

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise clamp to `[lo, hi]`.
#[derive(Clone)]
pub struct Clamp<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
}

impl<T: Scalar + Float, const N: usize> Clamp<T, N> {
    pub fn new(lo: T, hi: T) -> Self {
        Self { lo, hi }
    }
}

/// Runtime state for [`Clamp`]: the bounds plus the output buffer.
pub struct ClampState<T: Scalar, const N: usize> {
    lo: T,
    hi: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Clamp<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = ClampState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = ClampState {
            lo: self.lo,
            hi: self.hi,
            out: Array::zeros(x.extents()),
        };
        clamp_into(&mut state, x);
        state
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        clamp_into(state, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Clamp`]: clip `x` into `state.out`.
fn clamp_into<T: Scalar + Float, const N: usize>(
    state: &mut ClampState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let (lo, hi) = (state.lo, state.hi);
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let dst = state.out.data_mut();
    for i in 0..dst.len() {
        dst[i] = lo.max(hi.min(src[i]));
    }
}

/// Element-wise clamp into `[lo, hi]` (`NaN` passes through).
pub fn clamp<T: Scalar + Float, const N: usize>(lo: T, hi: T) -> Clamp<T, N> {
    Clamp::new(lo, hi)
}
