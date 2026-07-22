//! [`PctChange`] — the cross-tick one-step linear return.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Layout, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise one-step linear return: `input / input_prev - 1`.
#[derive(Clone)]
pub struct PctChange<T: Scalar + Float, const N: usize>(PhantomData<T>);

impl<T: Scalar + Float, const N: usize> PctChange<T, N> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T: Scalar + Float, const N: usize> Default for PctChange<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Runtime state for [`PctChange`]: the previous input (NaN-initialised) plus
/// the output buffer.
pub struct PctChangeState<T: Scalar + Float, const N: usize> {
    prev: Vec<T>,
    out: Array<T, N>,
}

impl<T: Scalar + Float, const N: usize> Operator for PctChange<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = PctChangeState<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        PctChangeState {
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
        let one = T::one();
        for i in 0..dst.len() {
            dst[i] = src[i] / state.prev[i] - one;
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

/// Element-wise one-step linear return: `x / x₋₁ − 1`. The `n`-tick
/// generalization over a live handle is [`growth`](super::super::formula::growth).
pub fn pct_change<T: Scalar + Float, const N: usize>() -> PctChange<T, N> {
    PctChange::new()
}
