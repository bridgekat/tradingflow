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
pub struct Event<T, I> {
    pub payload: Option<T>,
    pub stamp: Stamp<I>,
}

impl<T, I> Event<T, I> {
    /// An explicitly-stamped event: `payload` at timestamp `instant`.
    pub fn at(payload: T, instant: I) -> Self {
        Self {
            payload: Some(payload),
            stamp: Stamp::Instant(instant),
        }
    }

    /// An implicitly-stamped event: `payload` is stamped with the wall clock
    /// at receipt.
    pub fn now(payload: T) -> Self {
        Self {
            payload: Some(payload),
            stamp: Stamp::Now,
        }
    }

    /// A payload-less frontier advance: a promise that no event will arrive
    /// stamped strictly below `t`.
    pub fn frontier(instant: I) -> Self {
        Self {
            payload: None,
            stamp: Stamp::Instant(instant),
        }
    }
}
