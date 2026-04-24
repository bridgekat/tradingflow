//! Add operator — element-wise addition of two `Array<f64>`.

use crate::{Array, Input, Instant};

use crate::experimental::Operator;

/// Element-wise addition: `a + b`.
pub struct Add;

impl Add {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Add {
    type State = ();
    type Inputs = (Input<Array<f64>>, Input<Array<f64>>);
    type Output = Array<f64>;

    fn init(self, inputs: (&Array<f64>, &Array<f64>), _timestamp: Instant) -> ((), Array<f64>) {
        ((), Array::zeros(inputs.0.shape()))
    }

    fn compute(
        _state: &mut (),
        inputs: (&Array<f64>, &Array<f64>),
        output: &mut Array<f64>,
        _timestamp: Instant,
        _produced: (bool, bool),
    ) -> bool {
        let (a, b) = inputs;
        let dst = output.as_mut_slice();
        let src_a = a.as_slice();
        let src_b = b.as_slice();
        for i in 0..dst.len() {
            dst[i] = src_a[i] + src_b[i];
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = Array::from_vec(&[3], vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(&[3], vec![10.0, 20.0, 30.0]);
        let (mut s, mut o) = Add::new().init((&a, &b), Instant::MIN);
        Add::compute(&mut s, (&a, &b), &mut o, Instant::from_nanos(1), (false, false));
        assert_eq!(o.as_slice(), &[11.0, 22.0, 33.0]);
    }
}
