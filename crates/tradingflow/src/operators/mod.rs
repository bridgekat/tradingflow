//! `operators` — the operator library for the TradingFlow engine, built on
//! the engine. The engine/driver itself is [`Scenario`](crate::Scenario) /
//! [`Session`](crate::Session) (see [`ingest`](crate::ingest)). There is no
//! builder extension trait: every node is a `push` of one of the free
//! constructors below.
//!
//! # Design
//!
//! * Operators implement [`Operator`](crate::graph::Operator) (notify-gated compute /
//!   passthrough) or [`Segment`](crate::graph::Segment) (custom gating, e.g.
//!   [`Clocked`]) **directly** — no TradingFlow-side operator trait or bridge.
//!   **Every** `Array`-shaped edge carries a strided [`ArrayView`](crate::ArrayView)
//!   by value through `ArrayPort<T, N>` (and every `Series`-shaped edge a
//!   [`SeriesView`](crate::SeriesView) through `SeriesPort<T, N>`) — including
//!   source cells (an engine [`ViewSource`](crate::graph::ViewSource) lends its
//!   owned cell; see [`array_cell`]) and the Python host — so there are no
//!   owned↔view bridge operators. Output buffers live in each
//!   operator's `State` (an owned [`Array<T, N>`](crate::Array)) and `compute`
//!   lends a view of it. The `init == true` build call sizes/seeds buffers from
//!   the build-time input values without running per-tick side effects (see
//!   [`op`] for the conventions). Because operators are plain segments, they
//!   compose with the engine's combinators (`then`/`fork`/`par`) and the
//!   `segment!` macro.
//! * The engine's input-notification gating prunes the recompute cone: an
//!   [`Operator`](crate::graph::Operator)'s compute path fires iff ≥1 input
//!   notified (else its `passthrough` re-emits the previous output, un-notified).
//! * **The contract that makes this sound: *no-notify ⟹ output unchanged*.** See
//!   the [`op`] module conventions; it is a producer-side duty obeyed by every
//!   operator here.
//! * Time is ambient: every operator's `Context` is the [`Instant`](crate::Instant)
//!   the driver sets to the batch's event time before each `stabilize`, so
//!   `compute` is handed the timestamp (only operators that stamp event time
//!   read it, and it is never a graph dependency).
//! * **Every segment has a lowercase free constructor** — `percentile()`,
//!   `winsorize(p)`, `stack(axis)`, `benchmark(n, cash, adj)`, … — taking the
//!   operator's parameters and leaving `T` / `N` to be inferred *from the
//!   wiring* at [`segment`](crate::graph::Builder::segment). Prefer them to the
//!   inherent `Op::<T, N>::new(..)` forms, which need a turbofish at every call
//!   site. Constructors are what `segment!` applies with `@`, so a formula
//!   annotates only the segment's parameters and output interface — every
//!   operator in between infers `T` / `N` from the wiring.
//! * The **formula constructors** ([`ma`], [`lag`], [`change`], …) curry a
//!   private, bounded [`Record`] into the windowed operators, so a `segment!`
//!   formula over live array handles reads like the formula itself — retention
//!   sizing happens inside the constructor, and event time arrives through the
//!   graph context, so they take no clock. See the [`formula`] module docs for the
//!   private-record trade-off, the hoisted shared-record escape hatch, and the
//!   naming rule relating them to their `Series`-consuming primitives
//!   ([`rolling_mean`], [`ema_series`], [`lag_series`], …).

mod arith;
pub mod formula;
mod gating;
mod metrics;
mod num;
pub mod op;
mod reshape;
mod rolling;
mod stocks;
mod structural;
mod traders;
mod transform;

#[cfg(feature = "python")]
mod pyhost;

pub use op::{
    ArrayPort, ArrayPorts, ArrayValue, SeriesPort, SeriesPorts, SeriesValue, StripNotify,
    array_cell, series_cell,
};

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
    Cast, Id, ResampleClocked, ResampleView, Where, cast, id, keep_where, resample_clocked,
    resample_view,
};
pub use traders::{Benchmark, RandomTrader, SimpleTrader, benchmark, random_trader, simple_trader};
pub use transform::{
    Apply, ApplyInplace, Lag, Map, MapInplace, Select, SliceView, apply, apply_inplace, lag_series,
    map, map_inplace, select, select_along_axis, select_flat, slice_view,
};

#[cfg(feature = "python")]
pub use pyhost::{
    NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams, py_class_operator,
    py_class_operator_file, py_class_operator_source,
};

#[cfg(test)]
mod tests;
