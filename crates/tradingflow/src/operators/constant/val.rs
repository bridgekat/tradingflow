use crate::data::Instant;
use crate::graph::{Port, Segment, Val};

/// Holds a `Copy` value and re-emits it by value on every tick.
pub struct ConstVal<T: Copy + Send + Sync + 'static> {
    value: T,
}

impl<T: Copy + Send + Sync + 'static> ConstVal<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Copy + Send + Sync + 'static> Segment for ConstVal<T> {
    type Inputs = ();
    type Outputs = Port<Val<T>>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T) -> (bool, T) {
        (true, *state)
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &Instant) -> (bool, T) {
        (true, *state)
    }
}

/// A constant value cell.
pub fn const_val<T: Copy + Send + Sync + 'static>(value: T) -> ConstVal<T> {
    ConstVal::new(value)
}
