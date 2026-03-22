//! Record operator — records array values into a time series.

use crate::array::Array;
use crate::operator::Operator;
use crate::series::Series;
use crate::types::Scalar;

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
    ) -> bool {
        output.push(timestamp, inputs.0.as_slice());
        true
    }
}
