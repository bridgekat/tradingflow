use std::ops::RangeFull;

use crate::{Duration, Instant, Scalar, SeriesView};

/// A retention bound for [`SeriesView`]. This is a helper struct: the
/// [`Retention::trim_count`] method calculates the number of elements to be
/// trimmed from the start of a series view, based on the retention bound and
/// the current time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Retention {
    unbounded: bool,
    count: Option<usize>,
    duration: Option<Duration>,
}

impl Retention {
    /// Keeps all history.
    pub const fn unbounded() -> Self {
        Self {
            unbounded: true,
            count: None,
            duration: None,
        }
    }

    /// Keeps the most-recent `count` elements.
    pub const fn count(count: usize) -> Self {
        Self {
            unbounded: false,
            count: Some(count),
            duration: None,
        }
    }

    /// Keeps all elements within `duration` of the latest timestamp
    /// (exclusive).
    pub const fn duration(duration: Duration) -> Self {
        Self {
            unbounded: false,
            count: None,
            duration: Some(duration),
        }
    }

    /// Keeps the union of a `count` window and a `duration` window.
    pub const fn count_and_duration(count: usize, duration: Duration) -> Self {
        Self {
            unbounded: false,
            count: Some(count),
            duration: Some(duration),
        }
    }

    /// Calculates the number of elements to be trimmed from the start of a
    /// series view.
    pub fn trim_count<T: Scalar, const N: usize>(
        &self,
        series: SeriesView<'_, T, N>,
        now: &Instant,
    ) -> usize {
        if self.unbounded {
            return 0;
        }
        let range = series.range();
        let mut start = range.end;
        if let Some(c) = self.count {
            start = start.min(range.end.saturating_sub(c));
        }
        if let Some(d) = self.duration {
            let instants = series.instants();
            let cutoff = now.saturating_sub(d);
            let i = instants.partition_point(|&t| t <= cutoff);
            start = start.min(range.start + i);
        }
        start.saturating_sub(range.start)
    }
}

impl From<usize> for Retention {
    fn from(count: usize) -> Self {
        Self::count(count)
    }
}

impl From<Duration> for Retention {
    fn from(duration: Duration) -> Self {
        Self::duration(duration)
    }
}

impl From<RangeFull> for Retention {
    fn from(_: RangeFull) -> Self {
        Self::unbounded()
    }
}
