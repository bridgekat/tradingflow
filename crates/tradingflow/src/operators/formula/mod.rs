//! Formula constructors — curried, self-recording segments that make
//! [`segment!`](crate::segment) formulas read like the formulas themselves.
//!
//! Every windowed operator ([`RollingMean`](super::rolling::RollingMean), [`Lag`],
//! [`Ema`](super::rolling::Ema), …) consumes a
//! recorded [`Series`](tradingflow_data::Series), so writing `MA(x, 10)` over a live
//! array handle normally takes three steps:
//! register a [`Record`], size its [`Retention`](crate::data::Retention) to the consumer's look-back,
//! and wire the rolling operator. The constructors here fuse that chain into
//! one segment expression, so a signal like
//! `MA(x, 10) − MA(x, 5) > 0 AND NOT LAG(…, 1) > 0` is a two-line
//! [`segment!`](crate::segment) body:
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
//! series-consuming operators ([`RollingMean`](super::rolling::RollingMean), [`Lag`], …) against the shared
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
//! the window), and [`Retention`](crate::data::Retention) trims are amortized. Count-windowed
//! constructors therefore retain `n + 8` rows; the time-windowed [`ma_time`]
//! retains `window + 16 days`, which must exceed the longest gap between
//! consecutive events (weekends and holiday breaks for daily data — matching
//! the examples' long-standing slack). For exotic cadences, fall back to the
//! hoisted idiom with an explicit [`Retention`](crate::data::Retention).

mod change;
mod ema;
mod growth;
mod lag;
mod ma;
mod ma_time;
mod mstd;
mod msum;
mod mvar;

pub use change::*;
pub use ema::*;
pub use growth::*;
pub use lag::*;
pub use ma::*;
pub use ma_time::*;
pub use mstd::*;
pub use msum::*;
pub use mvar::*;

use super::structural::Record;
use super::transform::Lag;
use crate::data::Instant;
use crate::graph::cb::{Comp, Fork, Id};
use crate::ports::ArrayPort;

/// A private [`Record`] chained into a windowed consumer `O` — the shape every
/// self-recording constructor here returns.
pub type Windowed<T, const N: usize, O> = Comp<Record<T, N>, O>;

/// Fan-out of the live value and its private [`lag`], chained into a combiner
/// `O` — the [`change`] / [`growth`] shape.
pub type WithLagged<T, const N: usize, O> =
    Comp<Fork<Id<ArrayPort<T, N>, Instant>, Windowed<T, N, Lag<T, N>>>, O>;
