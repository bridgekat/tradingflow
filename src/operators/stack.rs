//! Stack operators — stacks N arrays along a new axis.
//!
//! * [`Stack`] — time-series semantics: copies all inputs on every
//!   trigger.
//! * [`NotifyStack`] — message-passing semantics: fills non-produced
//!   input slots with `NaN` (float-only).

use num_traits::Float;

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar, SliceProduced, SliceRefs};

/// Stack N homogeneous arrays along a new axis.
pub struct Stack<T: Scalar> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Stack<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`Stack`].
pub struct StackState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

impl<T: Scalar> Operator for Stack<T> {
    type State = StackState;
    type Inputs = [Input<Array<T>>];
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (StackState, Array<T>) {
        assert!(!inputs.is_empty(), "Stack requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = StackState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        (state, Array::zeros(&shape))
    }

    #[inline(always)]
    fn compute(
        state: &mut StackState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        super::concat::interleaved_copy(
            output,
            inputs.iter(),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

/// Stack N homogeneous float arrays along a new axis, using the `produced`
/// tree to distinguish freshly-produced inputs from stale ones.
///
/// On every compute, the output is first cleared to `NaN`, then only
/// the slots corresponding to inputs that produced in the current
/// flush cycle are copied in.  All other slots remain `NaN`.
///
/// This is the message-passing counterpart to [`Stack`] — where
/// `Stack` reads the latest value from every input (time-series
/// semantics), `NotifyStack` carries only values from inputs that
/// actually updated this cycle (message-passing semantics).
///
/// Typical use: combined with [`ForwardFill`](super::num::ForwardFill)
/// downstream to cleanly separate "freshly updated" from "last known"
/// state, e.g. for cross-sectional data whose sources update at
/// heterogeneous cadences (suspended stocks, multi-frequency sensors,
/// sparse event streams).
///
/// Float-only because `NaN` is used as the "no update" sentinel.
pub struct NotifyStack<T: Scalar + Float> {
    axis: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> NotifyStack<T> {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Runtime state for [`NotifyStack`].
pub struct NotifyStackState {
    outer_count: usize,
    chunk_size: usize,
    n_inputs: usize,
}

impl<T: Scalar + Float> Operator for NotifyStack<T> {
    type State = NotifyStackState;
    type Inputs = [Input<Array<T>>];
    type Output = Array<T>;

    fn init(
        self,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        _timestamp: Instant,
    ) -> (NotifyStackState, Array<T>) {
        assert!(!inputs.is_empty(), "NotifyStack requires at least one input");
        let first = inputs.get(0).shape();
        assert!(self.axis <= first.len(), "axis out of bounds");
        let state = NotifyStackState {
            outer_count: first[..self.axis].iter().product(),
            chunk_size: first[self.axis..].iter().product(),
            n_inputs: inputs.len(),
        };
        let mut shape = Vec::with_capacity(first.len() + 1);
        shape.extend_from_slice(&first[..self.axis]);
        shape.push(inputs.len());
        shape.extend_from_slice(&first[self.axis..]);
        let total: usize = shape.iter().product();
        (state, Array::from_vec(&shape, vec![T::nan(); total]))
    }

    #[inline(always)]
    fn compute(
        state: &mut NotifyStackState,
        inputs: SliceRefs<'_, Input<Array<T>>>,
        output: &mut Array<T>,
        _timestamp: Instant,
        produced: SliceProduced<'_, Input<Array<T>>>,
    ) -> bool {
        // Reset the entire output to NaN; then copy only produced inputs.
        for v in output.as_mut_slice().iter_mut() {
            *v = T::nan();
        }
        super::concat::interleaved_copy_selective(
            output,
            &inputs,
            (0..produced.len()).filter(|&i| produced.get(i)),
            state.n_inputs,
            state.outer_count,
            state.chunk_size,
        );
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;
    use crate::data::inputs::{empty_produced, produced_from_positions};
    use crate::operator::Operator;
    use crate::FlatRead;

    fn sp_empty(n: usize) -> SliceProduced<'static, Input<Array<f64>>> {
        empty_produced::<[Input<Array<f64>>]>(n)
    }

    fn sp_positions(positions: &[usize], n: usize) -> SliceProduced<'static, Input<Array<f64>>> {
        produced_from_positions::<[Input<Array<f64>>]>(positions, n)
    }

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

    // Two 2×3 matrices stacked along each possible axis.
    //
    // a = [[1,2,3],[4,5,6]]   b = [[7,8,9],[10,11,12]]
    //
    // flat(a) = [1,2,3,4,5,6]   flat(b) = [7,8,9,10,11,12]

    fn ab() -> (Array<f64>, Array<f64>) {
        let a = Array::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Array::from_vec(&[2, 3], vec![7., 8., 9., 10., 11., 12.]);
        (a, b)
    }

    #[test]
    fn matrix_axis0() {
        // shape [2,2,3]: new axis = batch
        // [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_empty(0),
        );
        assert_eq!(o.shape(), &[2, 2, 3]);
        assert_eq!(
            o.as_slice(),
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
        );
    }

    #[test]
    fn matrix_axis1() {
        // shape [2,2,3]: new axis between rows and cols
        // [[[1,2,3],[7,8,9]], [[4,5,6],[10,11,12]]]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(1).init(refs(&ptrs), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_empty(0),
        );
        assert_eq!(o.shape(), &[2, 2, 3]);
        assert_eq!(
            o.as_slice(),
            &[1., 2., 3., 7., 8., 9., 4., 5., 6., 10., 11., 12.]
        );
    }

    #[test]
    fn matrix_axis2() {
        // shape [2,3,2]: new axis = innermost
        // [[[1,7],[2,8],[3,9]], [[4,10],[5,11],[6,12]]]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = Stack::<f64>::new(2).init(refs(&ptrs), Instant::MIN);
        Stack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_empty(0),
        );
        assert_eq!(o.shape(), &[2, 3, 2]);
        assert_eq!(
            o.as_slice(),
            &[1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.]
        );
    }

    // ---- NotifyStack ----

    #[test]
    fn notify_both_produced_matches_stack() {
        // When every input produces, output is identical to plain Stack.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyStack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyStack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_positions(&[0, 1], 2),
        );
        assert_eq!(o.shape(), &[2, 2, 3]);
        assert_eq!(
            o.as_slice(),
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
        );
    }

    #[test]
    fn notify_only_first_produced_nans_second() {
        // Input 0 produced; input 1 did not — its slot is NaN.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyStack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyStack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_positions(&[0], 2),
        );
        assert_eq!(o.shape(), &[2, 2, 3]);
        let out = o.as_slice();
        assert_eq!(&out[0..6], &[1., 2., 3., 4., 5., 6.]);
        for v in &out[6..12] {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn notify_only_second_produced_nans_first() {
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyStack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyStack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_positions(&[1], 2),
        );
        let out = o.as_slice();
        for v in &out[0..6] {
            assert!(v.is_nan());
        }
        assert_eq!(&out[6..12], &[7., 8., 9., 10., 11., 12.]);
    }

    #[test]
    fn notify_no_produced_all_nan() {
        // Edge case: operator fires but no tracked input produced.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyStack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        NotifyStack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_empty(2),
        );
        for v in o.as_slice() {
            assert!(v.is_nan());
        }
    }

    #[test]
    fn notify_axis1_partial_update() {
        // Stack along axis 1 → shape [2,2,3]. Only input 1 updated.
        // Expected (with b's rows in slot 1, NaN in slot 0):
        //   [[[NaN,NaN,NaN],[7,8,9]], [[NaN,NaN,NaN],[10,11,12]]]
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (mut s, mut o) = NotifyStack::<f64>::new(1).init(refs(&ptrs), Instant::MIN);
        NotifyStack::compute(
            &mut s,
            refs(&ptrs),
            &mut o,
            Instant::from_nanos(1),
            sp_positions(&[1], 2),
        );
        assert_eq!(o.shape(), &[2, 2, 3]);
        let out = o.as_slice();
        for v in &out[0..3] {
            assert!(v.is_nan());
        }
        assert_eq!(&out[3..6], &[7., 8., 9.]);
        for v in &out[6..9] {
            assert!(v.is_nan());
        }
        assert_eq!(&out[9..12], &[10., 11., 12.]);
    }

    #[test]
    fn notify_init_output_is_nan() {
        // Initial output (before first compute) should be NaN throughout.
        let (a, b) = ab();
        let arrays: [&Array<f64>; 2] = [&a, &b];
        let ptrs = make_ptrs(&arrays);
        let (_, o) = NotifyStack::<f64>::new(0).init(refs(&ptrs), Instant::MIN);
        for v in o.as_slice() {
            assert!(v.is_nan());
        }
    }
}
