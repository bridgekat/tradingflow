//! Lag operator — ported from `src/operators/lag.rs`.

use super::super::data::{Array, Input, InputTypes, Instant, Scalar, Series};
use super::super::operator::Operator;

pub struct Lag<T: Scalar> {
    offset: usize,
    fill: T,
}

impl<T: Scalar> Lag<T> {
    pub fn new(offset: usize, fill: T) -> Self {
        Self { offset, fill }
    }
}

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
        let fill_arr = Array::from_vec(shape, vec![self.fill.clone(); stride]);
        let state = LagState {
            offset: self.offset,
            fill: self.fill,
        };
        (state, fill_arr)
    }

    fn compute(
        state: &mut LagState<T>,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _timestamp: Instant,
        _produced: <Self::Inputs as InputTypes>::Produced<'_>,
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
