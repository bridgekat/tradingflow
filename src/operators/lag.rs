//! Lag operator — outputs the value from N steps ago.

use crate::time::Instant;
use crate::{Notify, Operator, Scalar, Series};

/// Lag operator: outputs the value from `offset` steps ago.
///
/// If there are fewer than `offset + 1` values in the input series,
/// the output is filled with the provided `fill` value.
pub struct Lag<T: Scalar> {
    offset: usize,
    fill: T,
}

impl<T: Scalar> Lag<T> {
    pub fn new(offset: usize, fill: T) -> Self {
        Self { offset, fill }
    }
}

/// Runtime state: offset and fill value.
pub struct LagState<T: Scalar> {
    offset: usize,
    fill: T,
}

impl<T: Scalar> Operator for Lag<T> {
    type State = LagState<T>;
    type Inputs = (Series<T>,);
    type Output = Series<T>;

    fn init(self, inputs: (&Series<T>,), _timestamp: Instant) -> (LagState<T>, Series<T>) {
        let state = LagState {
            offset: self.offset,
            fill: self.fill,
        };
        (state, Series::new(inputs.0.shape()))
    }

    fn compute(
        state: &mut LagState<T>,
        inputs: (&Series<T>,),
        output: &mut Series<T>,
        timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let series = inputs.0;
        let len = series.len();
        if len > state.offset {
            output.push(timestamp, series.at(len - 1 - state.offset));
        } else {
            let stride = series.stride();
            let fill: Vec<T> = vec![state.fill.clone(); stride];
            output.push(timestamp, &fill);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Instant;

    fn ts(n: i64) -> Instant { Instant::from_nanos(n) }

    #[test]
    fn lag_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Lag::new(2, f64::NAN).init((&s,), Instant::MIN);

        s.push(ts(1), &[10.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(1), &Notify::new(&[], 0));
        assert!(out.last().unwrap()[0].is_nan());

        s.push(ts(2), &[20.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(2), &Notify::new(&[], 0));
        assert!(out.last().unwrap()[0].is_nan());

        s.push(ts(3), &[30.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(3), &Notify::new(&[], 0));
        assert_eq!(out.last().unwrap()[0], 10.0);

        s.push(ts(4), &[40.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(4), &Notify::new(&[], 0));
        assert_eq!(out.last().unwrap()[0], 20.0);
    }

    #[test]
    fn lag_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = Lag::new(1, f64::NAN).init((&s,), Instant::MIN);

        s.push(ts(1), &[1.0, 2.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(1), &Notify::new(&[], 0));
        assert!(out.last().unwrap()[0].is_nan());

        s.push(ts(2), &[3.0, 4.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(2), &Notify::new(&[], 0));
        assert_eq!(out.last().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn lag_integer_fill() {
        let mut s = Series::<i32>::new(&[]);
        let (mut state, mut out) = Lag::new(1, -1).init((&s,), Instant::MIN);

        s.push(ts(1), &[100]);
        Lag::compute(&mut state, (&s,), &mut out, ts(1), &Notify::new(&[], 0));
        assert_eq!(out.last().unwrap()[0], -1);

        s.push(ts(2), &[200]);
        Lag::compute(&mut state, (&s,), &mut out, ts(2), &Notify::new(&[], 0));
        assert_eq!(out.last().unwrap()[0], 100);
    }

    #[test]
    fn lag_timestamps() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Lag::new(1, 0.0).init((&s,), Instant::MIN);

        s.push(ts(100), &[1.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(100), &Notify::new(&[], 0));
        s.push(ts(200), &[2.0]);
        Lag::compute(&mut state, (&s,), &mut out, ts(200), &Notify::new(&[], 0));

        assert_eq!(out.timestamps(), &[ts(100), ts(200)]);
    }
}
