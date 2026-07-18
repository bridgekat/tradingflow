//! Formula constructors — curried, self-recording segments that make
//! [`segment!`](tradingflow_graph::segment) formulas read like the formulas themselves.
//!
//! Every windowed operator ([`RollingMean`], [`Lag`], [`Ema`], …) consumes a
//! recorded [`Series`](tradingflow_data::Series), so writing `MA(x, 10)` over a live
//! array handle normally takes three steps:
//! register a [`Record`], size its [`Retention`] to the consumer's look-back,
//! and wire the rolling operator. The constructors here fuse that chain into
//! one segment expression, so a signal like
//! `MA(x, 10) − MA(x, 5) > 0 AND NOT LAG(…, 1) > 0` is a two-line
//! [`segment!`](tradingflow_graph::segment) body:
//!
//! ```ignore
//! let signal = sc.segment(
//!     tradingflow::segment!(|x: ArrayPort<f64, 1>| -> ArrayPort<bool, 1> {
//!         let d = subtract() @ (ma(10) @ x, ma(5) @ x);
//!         and() @ (greater_than(0.0) @ d, not() @ (greater_than(0.0) @ lag(1) @ d))
//!     }),
//!     prices,
//! );
//! ```
//!
//! # Private records
//!
//! Each windowed constructor embeds its **own** bounded [`Record`], sized to
//! exactly its look-back (plus a small safety margin, below). This keeps call
//! sites free of retention arithmetic, at the cost of duplicating history when
//! several deep windows read the same stream: memory per use is
//! `window × element size`. That is negligible for the typical month-scale
//! window, but for a *year-deep* stream feeding many consumers, prefer the
//! hoisted idiom — record once via [`record_bounded`](super::structural::record_bounded) and wire the
//! series-consuming operators ([`RollingMean`], [`Lag`], …) against the shared
//! handle.
//!
//! # Naming
//!
//! Each constructor here consumes a **live array handle** and records
//! internally. Its `Series`-consuming primitive is reachable under the name of
//! the underlying operator: [`rolling_mean`](super::rolling::rolling_mean) /
//! [`rolling_sum`](super::rolling::rolling_sum) / [`rolling_variance`](super::rolling::rolling_variance)
//! for [`ma`] / [`msum`] / [`mvar`], and — where the names would otherwise
//! collide — [`ema_series`](super::rolling::ema_series) for [`ema`] and
//! [`lag_series`](super::transform::lag_series) for [`lag`]. Likewise the one-tick
//! specializations of [`change`] and [`growth`], which need no record at all,
//! are [`diff`](super::num::diff) and [`pct_change`](super::num::pct_change).
//!
//! # Time is not a parameter
//!
//! [`Record`] stamps rows with event time, but it is *handed* that time: the
//! event time is the graph-level context (`Context = Instant`), which the
//! driver sets before each `stabilize`. So no constructor here takes a clock,
//! and a `segment!` formula captures nothing at all.
//!
//! # Retention margins
//!
//! A private record must retain slightly *more* than the consumer's nominal
//! window: a sliding window reads the row it is about to evict (one row past
//! the window), and [`Retention`] trims are amortized. Count-windowed
//! constructors therefore retain `n + 8` rows; the time-windowed [`ma_time`]
//! retains `window + 16 days`, which must exceed the longest gap between
//! consecutive events (weekends and holiday breaks for daily data — matching
//! the examples' long-standing slack). For exotic cadences, fall back to the
//! hoisted idiom with an explicit [`Retention`].

use num_traits::Float;

use super::num::{BinaryFn, UnaryFn, divide, sqrt, subtract};
use super::rolling::{Ema, RollingMean, RollingSum, RollingVariance};
use super::structural::{Record, buffer};
use super::transform::Lag;
use crate::data::{Duration, Instant, Retention, Scalar};
use crate::graph::cb::{Comp, Fork, Id, Right};
use crate::ports::ArrayPort;

/// Extra time a private record retains beyond a time-delta window. Must exceed
/// the longest gap between consecutive events (weekends + holiday breaks for
/// daily data): a time window evicts rows up to one event *after* retention may
/// have trimmed them.
const TIME_MARGIN: Duration = Duration::from_days(16);

/// A private [`Record`] chained into a windowed consumer `O` — the shape every
/// self-recording constructor here returns.
pub type Windowed<T, const N: usize, O> = Comp<Record<T, N>, O>;

/// Fan-out of the live value and its private [`lag`], chained into a combiner
/// `O` — the [`change`] / [`growth`] shape.
pub type WithLagged<T, const N: usize, O> =
    Comp<Fork<Id<ArrayPort<T, N>, Instant>, Windowed<T, N, Lag<T, N>>>, O>;

/// Rolling mean of the last `n` ticks: `ma(n) @ x`. Self-recording;
/// `NaN` (un-notified) until `n` values have been seen.
pub fn ma<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingMean<T, N>> {
    Comp(buffer(n), RollingMean::count(n))
}

/// Rolling sum of the last `n` ticks: `msum(n) @ x`. Self-recording.
pub fn msum<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingSum<T, N>> {
    Comp(buffer(n), RollingSum::count(n))
}

/// Rolling population variance of the last `n` ticks: `mvar(n) @ x`.
/// Self-recording.
pub fn mvar<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, RollingVariance<T, N>> {
    Comp(buffer(n), RollingVariance::count(n))
}

/// Rolling standard deviation of the last `n` ticks (variance → square root,
/// fused): `mstd(n) @ x`. Self-recording.
pub fn mstd<T: Scalar + Float, const N: usize>(
    n: usize,
) -> Comp<Windowed<T, N, RollingVariance<T, N>>, UnaryFn<T, N>> {
    Comp(mvar(n), sqrt())
}

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

/// Exponential moving average with window-normalized weights (see [`Ema`]):
/// `ema(alpha, window) @ x`. Self-recording.
pub fn ema<T: Scalar + Float, const N: usize>(
    alpha: T,
    window: usize,
) -> Windowed<T, N, Ema<T, N>> {
    Comp(buffer(window), Ema::new(alpha, window))
}

/// The value from `n` ticks ago: `lag(n) @ x`. Self-recording; `NaN`
/// until more than `n` values have been seen.
pub fn lag<T: Scalar + Float, const N: usize>(n: usize) -> Windowed<T, N, Lag<T, N>> {
    lag_or(n, T::nan())
}

/// [`lag`] with an explicit fill value — for non-float scalars, or when a
/// missing lag should read as something other than `NaN`.
pub fn lag_or<T: Scalar, const N: usize>(n: usize, fill: T) -> Windowed<T, N, Lag<T, N>> {
    Comp(buffer(n), Lag::new(n, fill))
}

/// `n`-tick change `x − x₋ₙ` (the momentum shape): `change(n) @ x`.
/// Self-recording; `NaN` until the lag is available. The one-tick special
/// case, without a private record, is [`diff`](super::num::diff).
pub fn change<T: Scalar + Float, const N: usize>(n: usize) -> WithLagged<T, N, BinaryFn<T, N>> {
    Comp(Fork(Id::default(), lag(n)), subtract())
}

/// `n`-tick relative change `(x − x₋ₙ) / x₋ₙ` (the YoY-growth shape):
/// `growth(n) @ x`. The subtraction is kept (rather than reduced to
/// `x / x₋ₙ − 1`-style rank equivalents) so a negative base keeps its faithful
/// sign. Self-recording; `NaN` until the lag is available. The one-tick
/// special case, without a private record, is
/// [`pct_change`](super::num::pct_change).
#[allow(clippy::type_complexity)] // the combinator tree is the documentation
pub fn growth<T: Scalar + Float, const N: usize>(
    n: usize,
) -> WithLagged<
    T,
    N,
    Comp<Fork<BinaryFn<T, N>, Right<ArrayPort<T, N>, ArrayPort<T, N>, Instant>>, BinaryFn<T, N>>,
> {
    Comp(
        Fork(Id::default(), lag(n)),
        Comp(Fork(subtract(), Right::default()), divide()),
    )
}
