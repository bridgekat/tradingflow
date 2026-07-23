use num_traits::Float;

use crate::data::{Duration, Instant, Scalar};
use crate::graph::Segment;
use crate::graph::cb::Comp;
use crate::operators::rolling::RollingMean;
use crate::operators::series::buffer_duration;
use crate::ports::ArrayPort;

/// Rolling mean over a trailing **time** window (all values within `window` of
/// the latest tick): `ma_time(Duration::from_days(365)) @ x` — the TTM
/// idiom. Self-recording; the private record keeps exactly `window` under a
/// *delayed* trim, so the rows the sliding window evicts are still readable
/// on the tick they leave (see the module docs).
pub fn ma_time<T: Scalar + Float, const N: usize>(
    duration: Duration,
) -> impl Segment<Inputs = ArrayPort<T, N>, Outputs = ArrayPort<T, N>, Context = Instant> {
    Comp(buffer_duration(duration), RollingMean::time_delta(duration))
}
