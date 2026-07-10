//! `operators` — the operator library for the TradingFlow engine, built on
//! `flowgraph`. The engine/driver itself is `flowgraph::ingest`'s
//! [`Builder`](flowgraph::ingest::Builder) / [`Graph`](flowgraph::ingest::Graph)
//! (instantiated as [`Scenario`](crate::Scenario) / [`Session`](crate::Session)).
//! There is no builder extension trait: every node is a `push` of one of the
//! free constructors below.
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
//! * Time is threaded out-of-band through a shared [`EventTime`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//! * **Every segment has a lowercase free constructor** — `percentile()`,
//!   `winsorize(p)`, `stack(axis)`, `benchmark(n, cash, adj)`, … — taking the
//!   operator's parameters and leaving `T` / `N` to be inferred *from the
//!   wiring* at [`push`](flowgraph::typed::Builder::push). Prefer them to the
//!   inherent `Op::<T, N>::new(..)` forms, which need a turbofish at every call
//!   site. Constructors are what `segment!` applies with `@`, so a formula
//!   needs no type annotations beyond its parameters.
//! * The **formula constructors** ([`ma`], [`lag`], [`change`], …) curry a
//!   private, bounded [`Record`] into the windowed operators, so a `segment!`
//!   formula over live array handles reads like the formula itself — retention
//!   sizing and clock wiring happen inside the constructor (the clock is its
//!   explicit first parameter). See the [`formula`] module docs for the
//!   private-record trade-off, the hoisted shared-record escape hatch, and the
//!   naming rule relating them to their `Series`-consuming primitives
//!   ([`rolling_mean`], [`ema_series`], [`lag_series`], …).
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

#[cfg(feature = "python")]
mod pyhost;

pub use op::{ArrayValue, EventTime, StripNotify};

pub use arith::{
    Binary, BinaryFn, BinaryMap, Choose, Compare, CompareFn, Predicate, PredicateFn, Unary,
    UnaryFn, UnaryMap, abs, add, and, at_least, at_most, ceil, choose, divide, equal, equal_to,
    exp, exp2, floor, greater, greater_equal, greater_than, indicator, is_finite, is_nan, less,
    less_equal, less_than, log, log2, log10, max, min, multiply, negate, not, not_equal,
    not_equal_to, or, pow, recip, round, sign, sqrt, subtract, xor,
};
pub use formula::{
    Windowed, WithLagged, change, ema, growth, lag, lag_or, ma, ma_time, mstd, msum, mvar, record,
    record_bounded,
};
pub use gating::{Clocked, Count, Filter, Gate, Last, Record, clocked, count, filter, gate, last};
pub use metrics::{
    AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Turnover, Volatility, average_return,
    compound_return, drawdown, sharpe_ratio, turnover, volatility,
};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
    clamp, diff, fillna, forward_fill, gaussianize, pct_change, percentile, standardize, winsorize,
};
pub use reshape::{
    Concat, ConcatSync, Split, Stack, StackSync, concat, concat_sync, split, stack, stack_sync,
};
pub use rolling::{
    Accumulator, CovarianceAccumulator, Ema, MeanAccumulator, Rolling, RollingCovariance,
    RollingMean, RollingSum, RollingVariance, SumAccumulator, VarianceAccumulator, Window,
    ema_series, rolling, rolling_covariance, rolling_mean, rolling_sum, rolling_variance,
};
pub use stocks::{Annualize, ForwardAdjust, annualize, forward_adjust};
pub use structural::{
    Cast, Id, Resample, ResampleClocked, ResampleView, Where, cast, id, keep_where, resample,
    resample_clocked, resample_view,
};
pub use traders::{Benchmark, RandomTrader, SimpleTrader, benchmark, random_trader, simple_trader};
pub use transform::{
    Apply, ApplyInplace, AsView, DerefArrayView, Lag, Map, MapInplace, Own, RefArrayView, Select,
    SliceView, apply, apply_inplace, as_view, deref_array_view, lag_series, map, map_inplace, own,
    ref_array_view, ref_array_views, select, select_along_axis, select_flat, slice_view,
};

#[cfg(feature = "python")]
pub use pyhost::{
    NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams, py_class_operator,
    py_class_operator_file, py_class_operator_source,
};

#[cfg(test)]
mod tests;
