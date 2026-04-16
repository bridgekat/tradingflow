//! Lag operator — outputs the value from N steps ago.

use crate::{Array, Input, Instant, Notify, Operator, Scalar, Series};

/// Lag operator: reads a `Series<T>` and emits the element from `offset`
/// steps ago as an `Array<T>`.
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
    type Inputs = Input<Series<T>>;
    type Output = Array<T>;

    fn init(self, inputs: &Series<T>, _timestamp: Instant) -> (LagState<T>, Array<T>) {
        let shape = inputs.shape();
        let stride: usize = shape.iter().product();
        let fill = Array::from_vec(shape, vec![self.fill.clone(); stride]);
        let state = LagState {
            offset: self.offset,
            fill: self.fill,
        };
        (state, fill)
    }

    fn compute(
        state: &mut LagState<T>,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _notify: &Notify<'_>,
    ) -> bool {
        let series = inputs;
        let len = series.len();
        let dst = output.as_mut_slice();
        if len > state.offset {
            dst.clone_from_slice(series.at(len - 1 - state.offset));
        } else {
            dst.fill(state.fill.clone());
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Instant;

    fn ts(n: i64) -> Instant {
        Instant::from_nanos(n)
    }

    #[test]
    fn lag_basic() {
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Lag::new(2, f64::NAN).init(&s, Instant::MIN);
        assert!(out[0].is_nan());

        s.push(ts(1), &[10.0]);
        Lag::compute(&mut state, &s, &mut out, ts(1), &Notify::new(&[], 0));
        assert!(out[0].is_nan());

        s.push(ts(2), &[20.0]);
        Lag::compute(&mut state, &s, &mut out, ts(2), &Notify::new(&[], 0));
        assert!(out[0].is_nan());

        s.push(ts(3), &[30.0]);
        Lag::compute(&mut state, &s, &mut out, ts(3), &Notify::new(&[], 0));
        assert_eq!(out[0], 10.0);

        s.push(ts(4), &[40.0]);
        Lag::compute(&mut state, &s, &mut out, ts(4), &Notify::new(&[], 0));
        assert_eq!(out[0], 20.0);
    }

    #[test]
    fn lag_vector() {
        let mut s = Series::<f64>::new(&[2]);
        let (mut state, mut out) = Lag::new(1, f64::NAN).init(&s, Instant::MIN);
        assert!(out.as_slice()[0].is_nan());
        assert!(out.as_slice()[1].is_nan());

        s.push(ts(1), &[1.0, 2.0]);
        Lag::compute(&mut state, &s, &mut out, ts(1), &Notify::new(&[], 0));
        assert!(out.as_slice()[0].is_nan());

        s.push(ts(2), &[3.0, 4.0]);
        Lag::compute(&mut state, &s, &mut out, ts(2), &Notify::new(&[], 0));
        assert_eq!(out.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn lag_integer_fill() {
        let mut s = Series::<i32>::new(&[]);
        let (mut state, mut out) = Lag::new(1, -1).init(&s, Instant::MIN);
        assert_eq!(out[0], -1);

        s.push(ts(1), &[100]);
        Lag::compute(&mut state, &s, &mut out, ts(1), &Notify::new(&[], 0));
        assert_eq!(out[0], -1);

        s.push(ts(2), &[200]);
        Lag::compute(&mut state, &s, &mut out, ts(2), &Notify::new(&[], 0));
        assert_eq!(out[0], 100);
    }

    #[test]
    fn lag_offset_zero() {
        // Offset 0 — always outputs the most recent value.
        let mut s = Series::<f64>::new(&[]);
        let (mut state, mut out) = Lag::new(0, 0.0).init(&s, Instant::MIN);

        s.push(ts(1), &[42.0]);
        Lag::compute(&mut state, &s, &mut out, ts(1), &Notify::new(&[], 0));
        assert_eq!(out[0], 42.0);

        s.push(ts(2), &[99.0]);
        Lag::compute(&mut state, &s, &mut out, ts(2), &Notify::new(&[], 0));
        assert_eq!(out[0], 99.0);
    }
}
