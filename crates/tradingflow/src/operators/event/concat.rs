use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar, array};
use crate::graph::{Operator, Segment};
use crate::ports::{ArrayPort, ArrayPorts};

/// Operator signature for [`concat_sync`], [`stack_sync`] etc.
pub struct Combine<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[bool], &[ArrayView<'_, T, N>]) + Send + 'static,
{
    init: I,
    update: F,
    _marker: PhantomData<fn() -> (T, U)>,
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Combine<T, N, U, M, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[bool], &[ArrayView<'_, T, N>]) + Send + 'static,
{
    pub fn new(init: I, update: F) -> Self {
        Self {
            init,
            update,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar, const N: usize, U: Scalar, const M: usize, I, F> Operator
    for Combine<T, N, U, M, I, F>
where
    I: FnOnce(&[ArrayView<'_, T, N>]) -> Array<U, M> + Send + 'static,
    F: FnMut(&mut Array<U, M>, &[bool], &[ArrayView<'_, T, N>]) + Send + 'static,
{
    type Inputs = ArrayPorts<T, N>;
    type Outputs = ArrayPort<U, M>;
    type Context = Instant;
    type State = (F, Array<U, M>);

    fn init(self, (_, views): (&[bool], &[ArrayView<'_, T, N>])) -> (F, Array<U, M>) {
        (self.update, (self.init)(views))
    }

    fn passthrough<'a, 'b: 'a>(
        _: (&'a [bool], &'a [ArrayView<'a, T, N>]),
        (_, out): &'b mut (F, Array<U, M>),
    ) -> (bool, ArrayView<'a, U, M>) {
        (false, out.view())
    }

    fn compute<'a, 'b: 'a>(
        (flags, views): (&'a [bool], &'a [ArrayView<'a, T, N>]),
        (update, out): &'b mut (F, Array<U, M>),
        _: &Instant,
    ) -> (bool, ArrayView<'a, U, M>) {
        update(out, flags, views);
        (true, out.view())
    }
}

/// Like [`crate::operators::array::concat`], but emits NaN for inputs that
/// are not notified at each tick. Typically applied to synchronized inputs.
pub fn concat_sync<T: Scalar + Float, const N: usize>(
    axis: usize,
) -> impl Segment<Inputs = ArrayPorts<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    let init = move |views: &[ArrayView<'_, T, N>]| array::concat(views, axis);
    let update = move |out: &mut Array<T, N>, flags: &[bool], views: &[ArrayView<'_, T, N>]| {
        concat_into(out.data_mut(), flags, views, axis);
    };
    Combine::new(init, update)
}

/// Like [`crate::operators::array::stack`], but emits NaN for inputs that
/// are not notified at each tick. Typically applied to synchronized inputs.
pub fn stack_sync<T: Scalar + Float, const N: usize, const M: usize>(
    axis: usize,
) -> impl Segment<Inputs = ArrayPorts<T, N>, Outputs = ArrayPort<T, M>, Context = Instant> {
    assert!(
        M == N + 1,
        "stack_sync: output ndim ({M}) must be input ndim ({N}) plus one"
    );
    let init = move |views: &[ArrayView<'_, T, N>]| array::stack(views, axis);
    let update = move |out: &mut Array<T, M>, flags: &[bool], views: &[ArrayView<'_, T, N>]| {
        stack_into(out.data_mut(), flags, views, axis);
    };
    Combine::new(init, update)
}

/// Like [`array::concat_into`], but emits NaN for inputs with `flag == false`.
fn concat_into<T: Scalar + Float, const N: usize>(
    out: &mut [T],
    flags: &[bool],
    views: &[ArrayView<T, N>],
    axis: usize,
) {
    let Some(first) = views.first().map(|v| v.extents()) else {
        return;
    };
    let outer_count: usize = first[..axis].iter().product();
    let inner_count: usize = first[axis + 1..].iter().product();
    let chunks: Vec<usize> = views
        .iter()
        .map(|v| v.extents()[axis] * inner_count)
        .collect();
    interleave_into(out, flags, views, outer_count, &chunks);
}

/// Like [`array::stack_into`], but emits NaN for inputs with `flag == false`.
fn stack_into<T: Scalar + Float, const N: usize>(
    out: &mut [T],
    flags: &[bool],
    views: &[ArrayView<T, N>],
    axis: usize,
) {
    let Some(first) = views.first().map(|v| v.extents()) else {
        return;
    };
    let outer_count: usize = first[..axis].iter().product();
    let inner_count: usize = first[axis..].iter().product();
    let chunks = vec![inner_count; views.len()];
    interleave_into(out, flags, views, outer_count, &chunks);
}

/// Row-major interleave: for each of the `outer_count` outer indices, writes
/// each view's next `chunks[i]` scalars in view order. Assumes each view holds
/// exactly `outer_count * chunks[i]` scalars and `out` holds their total.
fn interleave_into<T: Scalar + Float, const N: usize>(
    out: &mut [T],
    flags: &[bool],
    views: &[ArrayView<T, N>],
    outer_count: usize,
    chunks: &[usize],
) {
    let mats: Vec<_> = views.iter().map(|v| v.to_contiguous()).collect();
    let mut dst = 0;
    for outer in 0..outer_count {
        for ((src, &chunk), &flag) in mats.iter().zip(chunks).zip(flags) {
            if flag {
                out[dst..dst + chunk].clone_from_slice(&src[outer * chunk..(outer + 1) * chunk]);
            } else {
                out[dst..dst + chunk].fill(T::nan());
            }
            dst += chunk;
        }
    }
}
