//! [`Fillna`] — element-wise NaN replacement with a constant.

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise NaN replacement with a constant.
#[derive(Clone)]
pub struct Fillna<T: Scalar, const N: usize> {
    val: T,
}

impl<T: Scalar + Float, const N: usize> Fillna<T, N> {
    pub fn new(val: T) -> Self {
        Self { val }
    }
}

/// Runtime state for [`Fillna`]: the fill value plus the output buffer.
pub struct FillnaState<T: Scalar, const N: usize> {
    val: T,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Fillna<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = FillnaState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        let mut state = FillnaState {
            val: self.val,
            out: Array::zeros(x.extents()),
        };
        fillna_into(&mut state, x);
        state
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        fillna_into(state, x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The per-call body of [`Fillna`]: copy `x` into `state.out`, substituting the
/// fill value for every NaN.
fn fillna_into<T: Scalar + Float, const N: usize>(
    state: &mut FillnaState<T, N>,
    x: ArrayView<'_, T, N>,
) {
    let val = state.val;
    let xs = x.to_contiguous();
    let src: &[T] = &xs;
    let dst = state.out.data_mut();
    for i in 0..dst.len() {
        dst[i] = if src[i].is_nan() { val } else { src[i] };
    }
}

/// Replace every non-finite element with `val`.
pub fn fillna<T: Scalar + Float, const N: usize>(val: T) -> Fillna<T, N> {
    Fillna::new(val)
}
