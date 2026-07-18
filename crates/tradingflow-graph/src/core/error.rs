use std::any::TypeId;

/// Errors returned by [`Builder::push`](super::Builder::push).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Wrong number of input slots for the segment.
    InputArity { expected: usize, actual: usize },
    /// Wrong number of output slots for the segment.
    OutputArity { expected: usize, actual: usize },
    /// Input `place` names `expected`, but only `num_slots` slots exist.
    InputOutOfBounds {
        place: usize,
        slot: usize,
        num_slots: usize,
    },
    /// Input `place` (`slot`) has type `actual`; the segment wants `expected`.
    InputType {
        place: usize,
        slot: usize,
        expected: TypeId,
        actual: TypeId,
    },
}
