//! [`ForwardFill`] — carry the last finite observation forward across ticks.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Forward-fills NaN with the last valid observation (per element position).
#[derive(Clone)]
pub struct ForwardFill<T: Scalar + Float, const N: usize> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float, const N: usize> ForwardFill<T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T: Scalar + Float, const N: usize> Default for ForwardFill<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float, const N: usize> Operator for ForwardFill<T, N> {
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    // The output buffer doubles as the fill memory: cells keep their last
    // non-NaN value across ticks because the state persists.
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        Array::full(x.extents(), T::nan())
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let dst = out.data_mut();
        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i];
            }
        }
        (true, out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// Carry the last finite value forward across ticks.
pub fn forward_fill<T: Scalar + Float, const N: usize>() -> ForwardFill<T, N> {
    ForwardFill::new()
}
