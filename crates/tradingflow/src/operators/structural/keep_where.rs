//! `Where` — element-wise conditional passthrough.

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace with `fill`.
#[derive(Clone)]
pub struct Where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> {
    condition: F,
    fill: T,
    _phantom: PhantomData<T>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize> Where<T, F, N> {
    pub fn new(condition: F, fill: T) -> Self {
        Self {
            condition,
            fill,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Where`]: the predicate and fill plus the output buffer.
pub struct WhereState<T: Scalar, F, const N: usize> {
    condition: F,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, F: Fn(T) -> bool + Clone + Send + Sync + 'static, const N: usize> Operator
    for Where<T, F, N>
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = WhereState<T, F, N>;

    fn init(self, (_, x): (bool, ArrayView<'_, T, N>)) -> Self::State {
        WhereState {
            condition: self.condition,
            fill: self.fill,
            out: Array::zeros(x.extents()),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let xs = x.to_contiguous();
        let src: &[T] = &xs;
        let out = state.out.data_mut();
        for i in 0..out.len() {
            out[i] = if (state.condition)(src[i].clone()) {
                src[i].clone()
            } else {
                state.fill.clone()
            };
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Element-wise conditional: keep the value where `condition` holds, else
/// replace it with `fill`. (Named `keep_where` because `where` is a keyword.)
pub fn keep_where<T: Scalar, F: Fn(T) -> bool + Clone, const N: usize>(
    condition: F,
    fill: T,
) -> Where<T, F, N> {
    Where::new(condition, fill)
}
