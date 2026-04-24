//! Variadic message-passing concatenation — ported from
//! `src/operators/concat.rs::ConcatSync`.

use num_traits::Float;

use super::super::data::{Array, Input, Instant, Scalar, SliceProduced, SliceRefs};
use super::super::operator::Operator;

pub struct ConcatSync<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> ConcatSync<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct ConcatSyncState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

impl<T: Scalar + Float> Operator for ConcatSync<T> {
    type State = ConcatSyncState;
    type Inputs = [Input<Array<T>>];
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (ConcatSyncState, Array<T>) {
        assert!(!inputs.is_empty(), "ConcatSync requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = ConcatSyncState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        let total: usize = shape.iter().product();
        (state, Array::from_vec(&shape, vec![T::nan(); total]))
    }

    fn compute(
        state: &mut ConcatSyncState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        produced: SliceProduced<'_, Input<Array<T>>>,
    ) -> bool {
        // Reset entire output to NaN, then copy only produced inputs.
        for v in output.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        let out = output.as_mut_slice();
        let stride = state.n_inputs * state.chunk_size;
        for pos in (0..produced.len()).filter(|&i| produced.get(i)) {
            let src = inputs.get(pos).as_slice();
            for outer in 0..state.outer_count {
                let src_offset = outer * state.chunk_size;
                let dst_offset = outer * stride + pos * state.chunk_size;
                out[dst_offset..dst_offset + state.chunk_size]
                    .clone_from_slice(&src[src_offset..src_offset + state.chunk_size]);
            }
        }
        true
    }
}
