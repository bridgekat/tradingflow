//! Record operator — records array values into a time series.

use crate::{Array, Input, InputTypes, Instant, Operator, Scalar, Series};

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
    type Inputs = Input<Array<T>>;
    type Output = Series<T>;

    fn init(self, inputs: &Array<T>, _timestamp: Instant) -> ((), Series<T>) {
        ((), Series::new(inputs.shape()))
    }

    fn compute(
        _state: &mut (),
        inputs: &Array<T>,
        output: &mut Series<T>,
        timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        output.push(timestamp, inputs.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instant;
    use crate::operator::Operator;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn scalar() {
        let a = Array::scalar(10.0_f64);
        let (mut s, mut o) = Record::<f64>::new().init(&a, Instant::MIN);
        Record::compute(&mut s, &a, &mut o, ts(100), false);
        let mut a2 = a.clone();
        a2[0] = 20.0;
        Record::compute(&mut s, &a2, &mut o, ts(200), false);
        assert_eq!(o.len(), 2);
        assert_eq!(o.timestamps(), &[ts(100), ts(200)]);
        assert_eq!(o.values(), &[10.0, 20.0]);
    }

    #[test]
    fn vector() {
        let a = Array::from_vec(&[2], vec![1.0, 2.0_f64]);
        let (mut s, mut o) = Record::<f64>::new().init(&a, Instant::MIN);
        Record::compute(&mut s, &a, &mut o, ts(1), false);
        assert_eq!(o.len(), 1);
        assert_eq!(o.shape(), &[2]);
        assert_eq!(o.at(0), &[1.0, 2.0]);
    }

    #[test]
    fn init_empty_series() {
        let a = Array::from_vec(&[2, 3], vec![0.0_f64; 6]);
        let (_, o) = Record::<f64>::new().init(&a, Instant::MIN);
        assert_eq!(o.shape(), &[2, 3]);
        assert_eq!(o.len(), 0);
    }
}
