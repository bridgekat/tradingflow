/// An explicit timestamp, or an implicit wall-clock time.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stamp<I> {
    Instant(I),
    Now,
}

/// An event from a stream.
///
/// Each event carries a frontier timestamp (non-decreasing in [`Stamp`]
/// ordering), and optionally a payload at that timestamp.
pub struct Event<I, T> {
    pub stamp: Stamp<I>,
    pub payload: Option<T>,
}

impl<I, T> Event<I, T> {
    /// An explicitly-stamped event: `payload` at timestamp `t`.
    pub fn at(instant: I, payload: T) -> Self {
        Self {
            stamp: Stamp::Instant(instant),
            payload: Some(payload),
        }
    }

    /// An implicitly-stamped event: `payload` is stamped with the wall clock
    /// at receipt.
    pub fn now(payload: T) -> Self {
        Self {
            stamp: Stamp::Now,
            payload: Some(payload),
        }
    }

    /// A payload-less frontier advance: a promise that no event will arrive
    /// stamped strictly below `t`.
    pub fn frontier(instant: I) -> Self {
        Self {
            stamp: Stamp::Instant(instant),
            payload: None,
        }
    }
}
