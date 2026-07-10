//! `operators` — the operator library for the TradingFlow engine, built on
//! `flowgraph`. The engine/driver itself is `flowgraph::ingest`'s
//! [`Builder`](flowgraph::ingest::Builder) / [`Graph`](flowgraph::ingest::Graph)
//! (instantiated as [`Scenario`](crate::Scenario) / [`Session`](crate::Session);
//! domain registrars in [`ScenarioExt`](crate::ScenarioExt)).
//!
//! # Design
//!
//! * Operators implement [`flowgraph::typed::Operator`] (notify-gated compute /
//!   passthrough) or [`flowgraph::typed::Segment`] (custom gating, e.g.
//!   [`Clocked`]) **directly** — no TradingFlow-side operator trait or bridge.
//!   Array-shaped edges carry a strided [`ArrayView`](crate::ArrayView) by value
//!   through `ViewPort<ArrayValue<T, N>>`; output buffers live in each operator's
//!   `State` (an owned [`Array<T, N>`](crate::Array)) and `compute` lends a view
//!   of it. The `init == true` build call sizes/seeds buffers from the build-time
//!   input values without running per-tick side effects (see [`op`] for the
//!   conventions). Because operators are plain segments, they compose with
//!   `flowgraph`'s combinators (`then`/`fork`/`par`) and the `segment!` macro.
//! * The engine's input-notification gating prunes the recompute cone: an
//!   [`Operator`](flowgraph::typed::Operator)'s compute path fires iff ≥1 input
//!   notified (else its `passthrough` re-emits the previous output, un-notified).
//! * **The contract that makes this sound: *no-notify ⟹ output unchanged*.** See
//!   the [`op`] module conventions; it is a producer-side duty obeyed by every
//!   operator here.
//! * Time is threaded out-of-band through a shared [`Clock`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//! * The **formula constructors** ([`ma`], [`lag`], [`change`], …) curry a
//!   private, bounded [`Record`] into the windowed operators, so a `segment!`
//!   formula over live array handles reads like the formula itself — retention
//!   sizing and clock wiring happen inside the constructor (the clock is its
//!   explicit first parameter). See the [`formula`] module docs for the
//!   private-record trade-off and the hoisted shared-record escape hatch.
//!
//! [array-view-refactor] Submodules are re-enabled file-by-file as they migrate
//! to the const-rank `ArrayView` currency (stages 2–4). `op`/`arith`/`num` are
//! migrated; the rest are temporarily gated.

mod arith;
mod formula;
mod gating;
mod metrics;
mod num;
mod op;
mod reshape;
mod rolling;
mod stocks;
mod structural;
mod traders;
mod transform;

// [array-view-refactor] the Python bridge is migrated to the const-rank view
// currency in stage 5.
#[cfg(feature = "python")]
mod pyhost;
#[cfg(feature = "python")]
mod python;

pub use op::{ArrayValue, Clock, StripNotify};

pub use arith::{
    Binary, BinaryFn, BinaryMap, Choose, Compare, CompareFn, Predicate, PredicateFn, Unary,
    UnaryFn, UnaryMap, abs, add, and, at_least, at_most, ceil, divide, equal, equal_to, exp, exp2,
    floor, greater, greater_equal, greater_than, indicator, is_finite, is_nan, less, less_equal,
    less_than, log, log2, log10, max, min, multiply, negate, not, not_equal, not_equal_to, or, pow,
    recip, round, sign, sqrt, subtract, xor,
};
pub use formula::{
    WithLagged, Windowed, change, ema, lag, lag_or, ma, ma_time, mstd, msum, mvar, pct_change,
    record, record_bounded,
};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
};
pub use gating::{Clocked, Count, Filter, Gate, Last, Record};
pub use structural::{Cast, Id, Resample, ResampleClocked, ResampleView, Where};
pub use transform::{
    Apply, ApplyInplace, AsView, DerefArrayView, Lag, Map, MapInplace, Own, RefArrayView, Select,
    SliceView,
};
pub use reshape::{Concat, ConcatSync, Split, Stack, StackSync};
pub use metrics::{AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Turnover, Volatility};
pub use stocks::{Annualize, ForwardAdjust};
pub use rolling::{
    Accumulator, CovarianceAccumulator, Ema, MeanAccumulator, Rolling, RollingCovariance,
    RollingMean, RollingSum, RollingVariance, SumAccumulator, VarianceAccumulator, Window,
};
pub use traders::{Benchmark, RandomTrader, SimpleTrader};

// [array-view-refactor] the Python bridge re-exports, restored in stage 5.
#[cfg(feature = "python")]
pub use pyhost::{NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams};
#[cfg(feature = "python")]
pub use python::PyOperator;

#[cfg(test)]
mod tests;
