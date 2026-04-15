//! Drawdown from previous high since inception.

use std::marker::PhantomData;

use num_traits::Float;

use crate::data::Instant;
use crate::{Array, Input, Notify, Operator, Scalar};

/// Drawdown from the running maximum: `(current - max) / max`.
///
/// Always non-positive.  Zero when at a new high.
pub struct Drawdown<T: Scalar + Float> {
    _phantom: PhantomData<T>,
}

impl<T: Scalar + Float> Drawdown<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

pub struct DrawdownState<T: Scalar + Float> {
    running_max: T,
}

impl<T: Scalar + Float> Operator for Drawdown<T> {
    type State = DrawdownState<T>;
    type Inputs = (Input<Array<T>>,);
    type Output = Array<T>;

    fn init(self, _inputs: (&Array<T>,), _timestamp: Instant) -> (Self::State, Array<T>) {
        (
            DrawdownState {
                running_max: T::nan(),
            },
            Array::scalar(T::zero()),
        )
    }

    fn compute(
        state: &mut DrawdownState<T>,
        inputs: (&Array<T>,),
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let current = inputs.0[0];
        if current.is_nan() {
            return false;
        }

        if state.running_max.is_nan() || current > state.running_max {
            state.running_max = current;
        }

        output[0] = if state.running_max > T::zero() {
            (current - state.running_max) / state.running_max
        } else {
            T::zero()
        };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Array::scalar(0.0_f64);
        let (mut s, mut o) = Drawdown::new().init((&a,), Instant::from_nanos(0));

        let mut a = Array::scalar(100.0);
        Drawdown::compute(&mut s, (&a,), &mut o, Instant::from_nanos(1), &Notify::new(&[], 0));
        assert_eq!(o[0], 0.0); // first value = max

        a[0] = 120.0;
        Drawdown::compute(&mut s, (&a,), &mut o, Instant::from_nanos(2), &Notify::new(&[], 0));
        assert_eq!(o[0], 0.0); // new high

        a[0] = 90.0;
        Drawdown::compute(&mut s, (&a,), &mut o, Instant::from_nanos(3), &Notify::new(&[], 0));
        assert!((o[0] - (-0.25)).abs() < 1e-10); // (90-120)/120

        a[0] = 110.0;
        Drawdown::compute(&mut s, (&a,), &mut o, Instant::from_nanos(4), &Notify::new(&[], 0));
        // still below 120 high
        assert!((o[0] - (-1.0 / 12.0)).abs() < 1e-10);

        a[0] = 130.0;
        Drawdown::compute(&mut s, (&a,), &mut o, Instant::from_nanos(5), &Notify::new(&[], 0));
        assert_eq!(o[0], 0.0); // new high
    }
}
