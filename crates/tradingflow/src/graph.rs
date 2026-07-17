//! The graph-building vocabulary — the engine's typed layer, re-exported.
//!
//! TradingFlow strategies are computation graphs: a
//! [`Scenario`](crate::Scenario) derefs to the typed [`Builder`], results come
//! back through [`Handle`]s and view ports, and custom operators implement
//! [`Segment`] / [`Operator`] directly. This module makes `tradingflow` the
//! only dependency a strategy crate needs: everything the engine exports for
//! graph building is available (and documented) here, and
//! [`segment!`](crate::segment) is the facade form of the fusion macro.
//!
//! The engine itself is the sibling [`tradingflow_graph`] workspace crate —
//! kept separate so it builds, tests, and runs Miri against its own minimal
//! dependency set, and stays free of any notion of time (see
//! [`ingest`](crate::ingest) for where time lives). Its crate docs cover the
//! core concepts: segments, interfaces, notification flags, and the
//! graph-level context.
//!
//! The untyped layer underneath ([`core`]) is re-exported for completeness —
//! [`Pool`] is the one item of it that appears in everyday signatures
//! ([`Session::pool`](crate::Session::pool)).

pub use tradingflow_graph::core;
pub use tradingflow_graph::core::Pool;
pub use tradingflow_graph::typed::*;

#[doc(hidden)]
pub use tradingflow_graph::segment as __segment;

/// Arrow notation for fused [`Segment`](crate::graph::Segment) composition:
/// named wires instead of point-free `Comp`/`Fork` plumbing. Segments apply to
/// wires prefix-style (`let c = add() @ (a, b);`); applications nest inside
/// any wire expression (`add() @ (a, add() @ (a, b))`, chaining
/// right-associatively), each nesting desugaring to a fresh intermediate wire.
/// The whole expression becomes a single scheduled node; all plumbing is
/// zero-copy ref routing.
///
/// ```ignore
/// let signal = sc.push(
///     tradingflow::segment!(|x: ArrayPort<f64, 1>| -> ArrayPort<bool, 1> {
///         let d = subtract() @ (ma(10) @ x, ma(5) @ x);
///         and() @ (greater_than(0.0) @ d, not() @ (greater_than(0.0) @ lag(1) @ d))
///     }),
///     prices,
/// );
/// ```
///
/// This is the engine's `segment!` re-exported through the facade: the
/// wrapper pins the expansion's runtime path to [`graph`](crate::graph) (via
/// the macro's `@[path]` override), so callers need no engine dependency
/// of their own.
#[macro_export]
macro_rules! segment {
    ($($tt:tt)*) => {
        $crate::graph::__segment!(@[$crate::graph] $($tt)*)
    };
}
