use num_traits::Float;

use super::Windowed;
use crate::data::{Duration, Retention, Scalar};
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingMean;
use crate::operators::structural::Record;

/// Extra time a private record retains beyond a time-delta window. Must exceed
/// the longest gap between consecutive events (weekends + holiday breaks for
/// daily data): a time window evicts rows up to one event *after* retention may
/// have trimmed them.
const TIME_MARGIN: Duration = Duration::from_days(16);

/// Rolling mean over a trailing **time** window (all values within `window` of
/// the latest tick): `ma_time(Duration::from_days(365)) @ x` — the TTM
/// idiom. Self-recording; see the module docs for the retention margin.
pub fn ma_time<T: Scalar + Float, const N: usize>(
    window: Duration,
) -> Windowed<T, N, RollingMean<T, N>> {
    Comp(
        Record::with_retention(Retention::duration(window + TIME_MARGIN)),
        RollingMean::time_delta(window),
    )
}
