//! Forward-fill operator — fills NaN values with the last valid observation.

use num_traits::Float;

use crate::data::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Forward-fills NaN values element-wise.
///
/// For each element position, outputs the most recent non-NaN value seen
/// so far. If no valid value has been seen yet, outputs NaN.
pub struct ForwardFill<T: Scalar + Float> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar + Float> ForwardFill<T> {
    /// Create a new forward-fill operator.
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar + Float> Default for ForwardFill<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar + Float> Operator for ForwardFill<T> {
    type State = ();
    type Inputs = (Input<Array<T>>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: Instant) -> ((), Array<T>) {
        let shape = inputs.0.shape();
        let stride: usize = shape.iter().product();
        ((), Array::from_vec(shape, vec![T::nan(); stride]))
    }

    fn compute(
        _state: &mut (),
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let src = inputs.0.as_slice();
        let dst = output.as_mut_slice();

        for i in 0..dst.len() {
            if !src[i].is_nan() {
                dst[i] = src[i].clone();
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffill_basic() {
        let mut a = Array::scalar(0.0_f64);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&a,), Instant::MIN);

        a[0] = 10.0;
        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(out[0], 10.0);

        a[0] = f64::NAN;
        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert_eq!(out[0], 10.0); // forward-filled

        a[0] = 30.0;
        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(3), &Notify::new(&[], 0));
        assert_eq!(out[0], 30.0);
    }

    #[test]
    fn ffill_initial_nan() {
        let mut a = Array::scalar(f64::NAN);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&a,), Instant::MIN);

        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert!(out[0].is_nan()); // no valid value yet

        a[0] = 5.0;
        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert_eq!(out[0], 5.0);
    }

    #[test]
    fn ffill_vector() {
        let mut a = Array::from_vec(&[3], vec![1.0, f64::NAN, 3.0]);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&a,), Instant::MIN);

        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(out.as_slice()[0], 1.0);
        assert!(out.as_slice()[1].is_nan());
        assert_eq!(out.as_slice()[2], 3.0);

        a.assign(&[f64::NAN, 20.0, f64::NAN]);
        ForwardFill::compute(&mut state, (&a,), &mut out, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert_eq!(out.as_slice()[0], 1.0);  // ffill
        assert_eq!(out.as_slice()[1], 20.0); // new value
        assert_eq!(out.as_slice()[2], 3.0);  // ffill
    }
}
