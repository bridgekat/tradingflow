//! Concat operators — concatenate N arrays along an existing axis.
//!
//! * [`Concat`] — time-series semantics: copies all inputs on every
//!   trigger.
//! * [`NotifyConcat`] — message-passing semantics: fills non-produced
//!   input slots with `NaN` (float-only).

use num_traits::Float;

use crate::{Array, Input, Instant, Notify, Operator, Scalar, SliceRefs};

/// Concatenate N homogeneous arrays along an existing axis.
pub struct Concat<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Concat<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Concat`].
pub struct ConcatState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

impl<T: Scalar> Operator for Concat<T> {
    type State = ConcatState;
    type Inputs = [Input<Array<T>>];
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (ConcatState, Array<T>) {
        assert!(!inputs.is_empty(), "Concat requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = ConcatState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut ConcatState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        interleaved_copy(
            output,
            inputs.iter(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

/// Concatenate N homogeneous float arrays along an existing axis,
/// using [`Notify`] to distinguish freshly-produced inputs from stale
/// ones.
///
/// On every compute, the output is first cleared to `NaN`, then only
/// the slots corresponding to inputs that produced in the current
/// flush cycle are copied in.  All other slots remain `NaN`.
///
/// Message-passing counterpart to [`Concat`].  See
/// [`NotifyStack`](super::stack::NotifyStack) for the general
/// motivation.
///
/// Float-only because `NaN` is used as the "no update" sentinel.
pub struct NotifyConcat<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> NotifyConcat<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`NotifyConcat`].
pub struct NotifyConcatState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

impl<T: Scalar + Float> Operator for NotifyConcat<T> {
    type State = NotifyConcatState;
    type Inputs = [Input<Array<T>>];
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (NotifyConcatState, Array<T>) {
        assert!(!inputs.is_empty(), "NotifyConcat requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis < first.len(), "axis out of bounds");
        let state = NotifyConcatState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = first.to_vec();
        shape[self.axis] *= inputs.len();
        let total: usize = shape.iter().product();
        (state, Array::from_vec(&shape, vec![T::nan(); total]))
    }

    #[inline(always)]
    fn compute(
        state: &mut NotifyConcatState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        notify: &Notify<'_>,
    ) -> bool {
        // Reset the entire output to NaN; then copy only produced inputs.
        for v in output.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        interleaved_copy_selective(
            output,
            &inputs,
            notify.produced(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

/// Copy data from N input arrays into an output array with interleaved
/// outer × chunk layout.  Used by [`Concat`] and [`Stack`].
///
/// Iterates `inputs` exactly once — safe for single-pass iterators from
/// [`SliceRefs::iter`].
#[inline(always)]
pub(super) fn interleaved_copy<'a, T: Scalar>(
    output: &mut Array<T>,
    inputs: impl IntoIterator<Item = &'a Array<T>>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for (input_idx, arr) in inputs.into_iter().enumerate() {
        let src = arr.as_slice();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + input_idx * chunk_size;
            out[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

/// Copy only the specified input positions into the output, leaving the
/// rest of the output untouched.  Used by [`NotifyConcat`] and
/// [`NotifyStack`](super::stack::NotifyStack) to fill only the slots
/// corresponding to freshly-produced inputs.
///
/// The caller is expected to initialise the output (e.g. with NaN)
/// before calling this function.
#[inline(always)]
pub(super) fn interleaved_copy_selective<T: Scalar>(
    output: &mut Array<T>,
    inputs: &SliceRefs<'_, Input<Array<T>>>,
    positions: impl IntoIterator<Item = usize>,
    n_inputs: usize,
    outer_count: usize,
    chunk_size: usize,
) {
    let out = output.as_mut_slice();
    let stride = n_inputs * chunk_size;
    for pos in positions {
        let src = inputs.get(pos).as_slice();
        for outer in 0..outer_count {
            let src_offset = outer * chunk_size;
            let dst_offset = outer * stride + pos * chunk_size;
            out[dst_offset..dst_offset + chunk_size]
                .clone_from_slice(&src[src_offset..src_offset + chunk_size]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;
    use crate::{FlatRead, InputTypes};

    /// Build a flat pointer buffer for a slice of Array refs.
    fn make_ptrs<'a, T: Scalar>(arrays: &'a [&'a Array<T>]) -> Vec<*const u8> {
        arrays
            .iter()
            .map(|&a| a as *const Array<T> as *const u8)
            .collect()
    }

    fn refs<'a, T: Scalar>(ptrs: &'a [*const u8]) -> SliceRefs<'a, Input<Array<T>>> {
        let mut reader = FlatRead::new(ptrs);
        unsafe { <[Input<Array<T>>] as InputTypes>::refs_from_flat(&mut reader) }
    }

    // Two 2×3×2 arrays concatenated along each axis.
    //
    // a[i][j][k] = 1 + i*6 + j*2 + k   (values 1..12)
    // b[i][j][k] = 13 + i*6 + j*2 + k   (values 13..24)
    //
    // flat(a) = [1,2,3,4,5,6,7,8,9,10,11,12]
    // flat(b) = [13,14,15,16,17,18,19,20,21,22,23,24]

    fn ab() -> (Array<f64>, Array<f64>) {
        let a = Array::from_vec(&[2, 3, 2], (1..=12).map(|x| x as f64).collect());
        let b = Array::from_vec(&[2, 3, 2], (13..=24).map(|x| x as f64).collect());
        (a, b)
    }

    #[test]
    fn array3d_axis0() {
        // [2,3,2] concat [2,3,2] along axis 0 → [4,3,2]
        // Just sequential: all of a, then all of b.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Concat::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        Concat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
        assert_eq!(o.shape(), &[4, 3, 2]);
        let expected: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        assert_eq!(o.as_slice(), &expected[..]);
    }

    #[test]
    fn array3d_axis1() {
        // [2,3,2] concat [2,3,2] along axis 1 → [2,6,2]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Concat::<f64>::new(1).init(refs(&ptrs), Instant::MIN);
        Concat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
        assert_eq!(o.shape(), &[2, 6, 2]);
        assert_eq!(
            o.as_slice(),
            &[
                1., 2., 3., 4., 5., 6., 13., 14., 15., 16., 17., 18., 7., 8., 9., 10., 11., 12.,
                19., 20., 21., 22., 23., 24.
            ]
        );
    }

    #[test]
    fn array3d_axis2() {
        // [2,3,2] concat [2,3,2] along axis 2 → [2,3,4]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Concat::<f64>::new(2).init(refs(&ptrs), Instant::MIN);
        Concat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 0),
        );
        assert_eq!(o.shape(), &[2, 3, 4]);
        assert_eq!(
            o.as_slice(),
            &[
                1., 2., 13., 14., 3., 4., 15., 16., 5., 6., 17., 18., 7., 8., 19., 20., 9., 10.,
                21., 22., 11., 12., 23., 24.
            ]
        );
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn scalar_panics() {
        let a = Array::scalar(1.0_f64);
        let arrays: [&Array<f64>; 1] = [&a];
        let ptrs = make_ptrs(&arrays);
        Concat::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
    }

    // ---- NotifyConcat ----

    #[test]
    fn notify_both_produced_matches_concat() {
        // axis=0 concat of two [2,3,2] arrays → [4,3,2], same as plain Concat.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyConcat::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyConcat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[0, 1], 2),
        );
        assert_eq!(o.shape(), &[4, 3, 2]);
        let expected: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        assert_eq!(o.as_slice(), &expected[..]);
    }

    #[test]
    fn notify_only_first_produced() {
        // Input 1 slot should be NaN; input 0 slot should be a.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyConcat::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyConcat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[0], 2),
        );
        let out = o.as_slice();
        let expected_a: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        assert_eq!(&out[0..12], &expected_a[..]);
        for v in &out[12..24] {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn notify_axis1_partial() {
        // [2,3,2] concat [2,3,2] along axis 1 → [2,6,2]; only input 1 produced.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyConcat::<f64>::new(1).init(refs(&ptrs), Instant::MIN);
        NotifyConcat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[1], 2),
        );
        assert_eq!(o.shape(), &[2, 6, 2]);
        let out = o.as_slice();
        for v in &out[0..6] {
            assert!(v.is_nan());
        }
        assert_eq!(&out[6..12], &[13., 14., 15., 16., 17., 18.]);
        for v in &out[12..18] {
            assert!(v.is_nan());
        }
        assert_eq!(&out[18..24], &[19., 20., 21., 22., 23., 24.]);
    }

    #[test]
    fn notify_no_produced_all_nan() {
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyConcat::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyConcat::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            &Notify::new(&[], 2),
        );
        for v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
