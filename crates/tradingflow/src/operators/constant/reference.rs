use crate::data::Instant;
use crate::graph::{Port, Ref, Segment};

/// Homes a value and lends it by reference on every tick.
pub struct ConstRef<T: Send + Sync + 'static> {
    value: T,
}

impl<T: Send + Sync + 'static> ConstRef<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Send + Sync + 'static> Segment for ConstRef<T> {
    type Inputs = ();
    type Outputs = Port<Ref<T>>;
    type Context = Instant;
    type State = T;

    fn init(self, _: ()) -> Self::State {
        self.value
    }

    fn output<'a, 'b: 'a>(_: (), state: &'b mut T) -> (bool, &'a T) {
        (true, state)
    }

    fn compute<'a, 'b: 'a>(_: (), state: &'b mut T, _: &Instant) -> (bool, &'a T) {
        (true, state)
    }
}

/// A constant reference cell.
pub fn const_ref<T: Send + Sync + 'static>(value: T) -> ConstRef<T> {
    ConstRef::new(value)
}
