//! Record operator — records array values into a time series.

use crate::{Array, Notify, Operator, Scalar, Series};

/// Record an array stream into a time series.
///
/// Shape is inferred from the input at init time.
pub struct Record<T: Scalar> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> Record<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar> Default for Record<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Operator for Record<T> {
    type State = ();
    type Inputs = (Array<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Array<T>,), _timestamp: i64) -> ((), Series<T>) {
        ((), Series::new(inputs.0.shape()))
    }

    fn compute(
        _state: &mut (),
        inputs: (&Array<T>,),
        output: &mut Series<T>,
        timestamp: i64,
        _notify: &Notify<'_>,
    ) -> bool {
        output.push(timestamp, inputs.0.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Operator;

    #[test]
    fn scalar() {
        let a = Array::scalar(10.0_f64);
        let (mut s, mut o) = Record::<f64>::new().init((&a,), i64::MIN);
        Record::compute(&mut s, (&a,), &mut o, 100, &Notify::new(&[], 0));
        let mut a2 = a.clone();
        a2[0] = 20.0;
        Record::compute(&mut s, (&a2,), &mut o, 200, &Notify::new(&[], 0));
        assert_eq!(o.len(), 2);
        assert_eq!(o.timestamps(), &[100, 200]);
        assert_eq!(o.values(), &[10.0, 20.0]);
    }

    #[test]
    fn vector() {
        let a = Array::from_vec(&[2], vec![1.0, 2.0_f64]);
        let (mut s, mut o) = Record::<f64>::new().init((&a,), i64::MIN);
        Record::compute(&mut s, (&a,), &mut o, 1, &Notify::new(&[], 0));
        assert_eq!(o.len(), 1);
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.at(0), &[1.0, 2.0]);
    }

    #[test]
    fn init_empty_series() {
        let a = Array::from_vec(&[2, 3], vec![0.0_f64; 6]);
        let (_, o) = Record::<f64>::new().init((&a,), i64::MIN);
        assert_eq!(o.shape(), &[2, 3]);
        assert_eq!(o.len(), 0);
    }
}
