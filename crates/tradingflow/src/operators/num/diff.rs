//! [`Diff`] — the cross-tick first difference.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise first difference across ticks: `input - input_prev`.
#[derive(Clone)]
pub struct Diff<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> Diff<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for Diff<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`Diff`]: the previous input (NaN-initialised) plus the
/// output buffer.
pub struct DiffState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for Diff<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = DiffState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        DiffState {
            prev: vec![T::nan(); x.layout().len()],
            out: Array::full(x.extents(), T::nan()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = state.out.data_mut();
        for i in 0..dst.len() {
            dst[i] = src[i] - state.prev[i];
        }
        state.prev.copy_from_slice(src);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Element-wise first difference across ticks: `x − x₋₁`. The `n`-tick
/// generalization over a live handle is [`change`](super::super::formula::change).
pub fn diff<T: Scalar + Float, const N: usize>() -> Diff<T, N> {
    Diff::new()
}
