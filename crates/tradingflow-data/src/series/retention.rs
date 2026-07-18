//! [`Retention`] — how much history a [`Series`](super::Series) keeps.

use crate::time::Duration;

/// A retention bound for a [`Series`](super::Series): how much history to keep.
///
/// A bound is the **union** of its active constraints — an element is retained
/// if *either* it is among the most-recent `count` elements *or* its timestamp
/// is within `duration` of the latest. This lets a single record feed both a
/// count-windowed consumer (e.g. `Lag(244)`, `RollingMean::count(252)`) and a
/// time-windowed one (e.g. `RollingMean::time_delta(365d)`): set both, and the
/// larger window wins. The default (both `None`) is unbounded — nothing is
/// dropped.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Retention {
    /// Keep at least the most-recent `count` elements (`None` = no count bound).
    pub count: Option<usize>,
    /// Keep at least all elements within `duration` of the latest timestamp
    /// (`None` = no time bound).
    pub duration: Option<Duration>,
}

impl Retention {
    /// Unbounded retention: never drop anything (the default).
    pub const UNBOUNDED: Retention = Retention {
        count: None,
        duration: None,
    };

    /// Keep the most-recent `count` elements.
    pub fn count(count: usize) -> Self {
        Self {
            count: Some(count),
            duration: None,
        }
    }

    /// Keep all elements within `duration` of the latest timestamp.
    pub fn duration(duration: Duration) -> Self {
        Self {
            count: None,
            duration: Some(duration),
        }
    }

    /// Keep the union of a `count` window and a `duration` window.
    pub fn count_and_duration(count: usize, duration: Duration) -> Self {
        Self {
            count: Some(count),
            duration: Some(duration),
        }
    }

    /// Whether this bound retains everything (no trimming).
    pub fn is_unbounded(&self) -> bool {
        self.count.is_none() && self.duration.is_none()
    }
}
