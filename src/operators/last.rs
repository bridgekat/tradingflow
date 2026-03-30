//! Last operator — extracts the most recent element from a Series.

use crate::{Array, Operator, Scalar, Series};

/// Extract the most recent element from a `Series<T>` as an `Array<T>`.
///
/// If the series is empty, the output is filled with the provided `default` value.
///
/// Two-sided inverse of [`Record`](super::Record):
/// - `Last(Record(x))` recovers the latest array value.
/// - `Record(Last(s)).last()` equals `s.last()`.
pub struct Last<T: Scalar> {
    fill: T,
}

impl<T: Scalar> Last<T> {
    pub fn new(default: T) -> Self {
        Self { fill: default }
    }
}

impl<T: Scalar> Operator for Last<T> {
    type State = T;
    type Inputs = (Series<T>,);
    type Output = Array<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: i64) -> (T, Array<T>) {
        let shape = inputs.0.shape();
        let out = if let Some(last) = inputs.0.last() {
            Array::from_vec(shape, last.to_vec())
        } else {
            Array::full(shape, self.fill.clone())
        };
        (self.fill, out)
    }

    fn compute(
        state: &mut T,
        inputs: (&Series<T>,),
        output: &mut Array<T>,
        _timestamp: i64,
    ) -> bool {
        if let Some(last) = inputs.0.last() {
            output.as_slice_mut().clone_from_slice(last);
        } else {
            let out = output.as_slice_mut();
            for v in out.iter_mut() {
                *v = state.clone();
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_basic() {
        let mut s = Series::<f64>::new(&[2]);
        s.push(100, &[1.0, 2.0]);
        s.push(200, &[3.0, 4.0]);

        let (mut state, mut out) = Last::new(0.0).init((&s,), i64::MIN);
        assert_eq!(out.as_slice(), &[3.0, 4.0]);

        s.push(300, &[5.0, 6.0]);
        Last::compute(&mut state, (&s,), &mut out, 300);
        assert_eq!(out.as_slice(), &[5.0, 6.0]);
    }

    #[test]
    fn last_empty_series() {
        let s = Series::<f64>::new(&[3]);
        let (mut state, mut out) = Last::new(f64::NAN).init((&s,), i64::MIN);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());
        assert!(out[2].is_nan());

        // Still empty on compute
        Last::compute(&mut state, (&s,), &mut out, 1);
        assert!(out[0].is_nan());
    }

    #[test]
    fn last_record_roundtrip() {
        use crate::operators::Record;

        // Last(Record(x)) == x for each of 10 elements
        let mut a = Array::scalar(0.0_f64);
        let (mut rec_s, mut series) = Record::new().init((&a,), i64::MIN);
        let (mut last_s, mut out) = Last::new(0.0).init((&series,), i64::MIN);

        for i in 1..=10 {
            let v = i as f64 * 7.0;
            a[0] = v;
            Record::compute(&mut rec_s, (&a,), &mut series, i);
            Last::compute(&mut last_s, (&series,), &mut out, i);
            assert_eq!(out[0], v, "mismatch at step {i}");
        }
    }

    #[test]
    fn record_last_roundtrip() {
        use crate::operators::Record;

        // Record(Last(x)) == x for each of 10 elements
        let mut s = Series::new(&[]);
        let (mut last_a, mut arr) = Last::new(0.0).init((&s,), i64::MIN);
        let (mut rec_a, mut out) = Record::new().init((&arr,), i64::MIN);

        for i in 1..=10 {
            let v = i as f64 * 7.0;
            s.push(i, &[v]);
            Last::compute(&mut last_a, (&s,), &mut arr, i);
            Record::compute(&mut rec_a, (&arr,), &mut out, i);
            assert_eq!(s.len(), out.len(), "mismatch at step {i}");
            for j in 0..s.len() {
                assert_eq!(out.at(j), s.at(j), "mismatch at step {i}, index {j}");
            }
        }
    }

    #[test]
    fn last_scalar() {
        let mut s = Series::<f64>::new(&[]);
        s.push(1, &[42.0]);
        let (mut state, mut out) = Last::new(0.0).init((&s,), i64::MIN);
        Last::compute(&mut state, (&s,), &mut out, 1);
        assert_eq!(out[0], 42.0);
    }
}
