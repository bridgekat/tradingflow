//! `Cast` — element-wise scalar type conversion.

use std::marker::PhantomData;

use num_traits::AsPrimitive;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise type conversion `ArrayView<S, N> → Array<T, N>` via
/// `AsPrimitive`.
#[derive(Clone)]
pub struct Cast<S: Scalar, T: Scalar, const N: usize> {
    _phantom: PhantomData<(S, T)>,
}

impl<S: Scalar, T: Scalar, const N: usize> Cast<S, T, N> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Scalar, T: Scalar, const N: usize> Default for Cast<S, T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T, const N: usize> Operator for Cast<S, T, N>
where
    S: Scalar + Copy + AsPrimitive<T>,
    T: Scalar + Copy + 'static,
{
    type Inputs = ArrayPort<S, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = Array<T, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, S, N>)) -> Self::State {
        Array::zeros(x.extents())
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, S, N>),
        out: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[S] = &xs;
        let dst = out.data_mut();
        for i in 0..dst.len() {
            dst[i] = src[i].as_();
        }
        (true, out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, S, N>),
        out: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, out.view())
    }
}

/// Element-wise scalar cast `S -> T`.
pub fn cast<S: Scalar, T: Scalar, const N: usize>() -> Cast<S, T, N> {
    Cast::new()
}
