//! Identity operator — passes input through unchanged.

use crate::{Input, Instant};

use crate::experimental::Operator;

/// Identity operator: clones input to output.
pub struct Id;

impl Id {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

impl Operator for Id {
    type State = ();
    type Inputs = Input<f64>;
    type Output = f64;

    fn init(self, inputs: &f64, _timestamp: Instant) -> ((), f64) {
        ((), *inputs)
    }

    fn compute(
        _state: &mut (),
        inputs: &f64,
        output: &mut f64,
        _timestamp: Instant,
        _produced: bool,
    ) -> bool {
        *output = *inputs;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let a = 42.0_f64;
        let (mut s, mut o) = Id::new().init(&a, Instant::MIN);
        Id::compute(&mut s, &42.0, &mut o, Instant::from_nanos(1), false);
        assert_eq!(o, 42.0);
    }
}
