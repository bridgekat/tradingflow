//! `operators` — the operator library for the TradingFlow engine, built on
//! the engine. The engine/driver itself is [`Scenario`](crate::Scenario) /
//! [`Session`](crate::Session) (see [`ingest`](crate::ingest)). There is no
//! builder extension trait: every node is a `push` of one of the free
//! constructors below.
//!
//! # Design
//!
//! * Operators implement [`Operator`](tradingflow_graph::typed::Operator) (notify-gated compute /
//!   passthrough) or [`Segment`](tradingflow_graph::typed::Segment) (custom gating, e.g.
//!   [`Clocked`](structural::Clocked)) **directly** — no TradingFlow-side operator trait or bridge.
//!   Every edge carries a borrowed view by value through the
//!   [`ports`](crate::ports) currency — including source cells (an engine
//!   [`ViewSource`](tradingflow_graph::typed::ViewSource) lends its owned cell; see
//!   [`array_cell`](constant::array_cell)) and the Python host — so there are no
//!   owned↔view bridge operators. Output buffers live in each
//!   operator's `State` (an owned [`Array<T, N>`](tradingflow_data::Array)) and `compute`
//!   lends a view of it. The `init == true` build call sizes/seeds buffers from
//!   the build-time input values without running per-tick side effects. Because
//!   operators are plain segments, they compose with the engine's combinators
//!   (`then`/`fork`/`par`) and the `segment!` macro.
//! * The engine's input-notification gating prunes the recompute cone: an
//!   [`Operator`](tradingflow_graph::typed::Operator)'s compute path fires iff ≥1 input
//!   notified (else its `passthrough` re-emits the previous output, un-notified).
//! * **The contract that makes this sound: *no-notify ⟹ output unchanged*** — a
//!   producer-side duty obeyed by every operator here.
//! * Time is ambient: every operator's `Context` is the [`Instant`](tradingflow_data::Instant)
//!   the driver sets to the batch's event time before each `stabilize`, so
//!   `compute` is handed the timestamp (only operators that stamp event time
//!   read it, and it is never a graph dependency).
//! * **Every segment has a lowercase free constructor** — `percentile()`,
//!   `winsorize(p)`, `stack(axis)`, `benchmark(n, cash, adj)`, … — taking the
//!   operator's parameters and leaving `T` / `N` to be inferred *from the
//!   wiring* at [`segment`](tradingflow_graph::typed::Builder::segment). Prefer them to the
//!   inherent `Op::<T, N>::new(..)` forms, which need a turbofish at every call
//!   site. Constructors are what `segment!` applies with `@`, so a formula
//!   annotates only the segment's parameters and output interface — every
//!   operator in between infers `T` / `N` from the wiring.
//! * The **formula constructors** ([`ma`](formula::ma), [`lag`](formula::lag),
//!   [`change`](formula::change), …) curry a private, bounded
//!   [`Record`](structural::Record) into the windowed operators, so a `segment!`
//!   formula over live array handles reads like the formula itself — retention
//!   sizing happens inside the constructor, and event time arrives through the
//!   graph context, so they take no clock. See the [`formula`] module docs for the
//!   private-record trade-off, the hoisted shared-record escape hatch, and the
//!   naming rule relating them to their `Series`-consuming primitives
//!   ([`rolling_mean`](rolling::rolling_mean), [`ema_series`](rolling::ema_series),
//!   [`lag_series`](transform::lag_series), …).

pub mod constant;
pub mod formula;
pub mod metrics;
pub mod num;
pub mod rolling;
pub mod stocks;
pub mod structural;
pub mod traders;
pub mod transform;

#[cfg(feature = "python")]
mod pyhost;

#[cfg(feature = "python")]
pub use pyhost::{
    NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams, py_class_operator,
    py_class_operator_file, py_class_operator_source,
};

#[cfg(test)]
mod tests;
