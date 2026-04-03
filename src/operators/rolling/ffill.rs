//! Forward-fill operator — fills NaN values with the last valid observation.

use num_traits::Float;

use crate::{Array, Notify, Operator, Scalar, Series};

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
    type State = Vec<T>;
    type Inputs = (Series<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (Vec<T>, Array<T>) {
        let stride = inputs.0.stride();
        let last_valid = vec![T::nan(); stride];
        (last_valid, Array::zeros(inputs.0.shape()))
    }

    fn compute(
        state: &mut Vec<T>,
        inputs: (&Series<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        let series = inputs.0;
        let row = series.last().unwrap();
        let stride = state.len();
        let out = output.as_mut_slice();

        for i in 0..stride {
            if !row[i].is_nan() {
                state[i] = row[i];
            }
            out[i] = state[i];
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffill_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&s,), i64::MIN);

        s.push(1, &[10.0]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 10.0);

        s.push(2, &[f64::NAN]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 10.0); // forward-filled

        s.push(3, &[30.0]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 3, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 30.0);
    }

    #[test]
    fn ffill_initial_nan() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&s,), i64::MIN);

        s.push(1, &[f64::NAN]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], &[]));
        assert!(out.as_slice()[0].is_nan()); // no valid value yet

        s.push(2, &[5.0]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 5.0);
    }

    #[test]
    fn ffill_vector() {
        let mut s = Series::<f64>::new(&[3]);
        let (mut state, mut out) = ForwardFill::<f64>::new().init((&s,), i64::MIN);

        s.push(1, &[1.0, f64::NAN, 3.0]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 1, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 1.0);
        assert!(out.as_slice()[1].is_nan());
        assert_eq!(out.as_slice()[2], 3.0);

        // Element 1 gets filled from element 1's first valid value
        s.push(2, &[f64::NAN, 20.0, f64::NAN]);
        ForwardFill::compute(&mut state, (&s,), &mut out, 2, &Notify::new(&[], &[]));
        assert_eq!(out.as_slice()[0], 1.0); // ffill
        assert_eq!(out.as_slice()[1], 20.0); // new value
        assert_eq!(out.as_slice()[2], 3.0); // ffill
    }
}
