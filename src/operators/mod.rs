//! `operators` — the operator library for the TradingFlow engine, built on
//! `flowgraph`. The engine/driver itself ([`Scenario`](crate::Scenario) /
//! [`Session`](crate::Session)) lives in [`scenario`](crate::scenario).
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
//!
//! [array-view-refactor] Submodules are re-enabled file-by-file as they migrate
//! to the const-rank `ArrayView` currency (stages 2–4). `op`/`arith`/`num` are
//! migrated; the rest are temporarily gated.

mod arith;
mod gating;
mod num;
mod op;
mod structural;

#[cfg(any())]
mod metrics;
#[cfg(any())]
#[cfg(feature = "python")]
mod pyhost;
#[cfg(any())]
#[cfg(feature = "python")]
mod python;
#[cfg(any())]
mod reshape;
#[cfg(any())]
mod rolling;
#[cfg(any())]
mod stocks;
#[cfg(any())]
mod structural;
#[cfg(any())]
mod traders;
#[cfg(any())]
mod transform;

pub use op::{ArrayValue, Clock, StripNotify};

pub use arith::{
    Binary, BinaryFn, Unary, UnaryFn, abs, add, ceil, divide, exp, exp2, floor, log, log2, log10,
    max, min, multiply, negate, pow, recip, round, sign, sqrt, subtract,
};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
};
pub use gating::{Clocked, Count, Filter, Gate, Last, Record};
pub use structural::{Cast, Id, Resample, Where};

// [array-view-refactor] re-export blocks for gated families, restored per stage:
// pub use metrics::{AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Turnover, Volatility};
// pub use reshape::{Concat, ConcatSync, Split, Stack, StackSync, StackSyncView, StackView};
// pub use stocks::{Annualize, AnnualizeView, ForwardAdjust, ForwardAdjustViewDiv};
// pub use transform::{Apply, ApplyInplace, Lag, Map, MapInplace, Select, SelectView, SliceView};
// pub use rolling::{...};
// pub use structural::{Cast, Id, Resample, Where};
// pub use traders::{Benchmark, RandomTrader, SimpleTrader};
// #[cfg(feature = "python")]
// pub use pyhost::{NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams};
// #[cfg(feature = "python")]
// pub use python::PyOperator;

#[cfg(any())]
#[cfg(test)]
mod tests;
