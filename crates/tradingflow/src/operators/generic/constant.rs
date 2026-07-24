use crate::data::Instant;
use crate::graph::{Port, Ref, Segment, Val};

/// Operator signature for [`const_val`] etc.
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

/// Operator signature for [`const_ref`] etc.
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

/// A constant value cell.
pub fn const_val<T: Copy + Send + Sync + 'static>(
    value: T,
) -> impl Segment<Inputs = (), Outputs = Port<Val<T>>, Context = Instant, State = T> {
    ConstVal::new(value)
}

/// A constant reference cell.
pub fn const_ref<T: Send + Sync + 'static>(
    value: T,
) -> impl Segment<Inputs = (), Outputs = Port<Ref<T>>, Context = Instant, State = T> {
    ConstRef::new(value)
}
