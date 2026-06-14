//! `operators` — the operator library for the TradingFlow engine, built on
//! `flowgraph`. The engine/driver itself ([`Scenario`](crate::Scenario) /
//! [`Session`](crate::Session)) lives in [`scenario`](crate::scenario).
//!
//! # Design
//!
//! * Operators implement [`flowgraph::typed::Operator`] (notify-gated compute /
//!   passthrough) or [`flowgraph::typed::Segment`] (custom gating, e.g.
//!   [`Clocked`]) **directly** — no TradingFlow-side operator trait or bridge.
//!   Output buffers live in each operator's `State`; `compute` returns
//!   `(notify, &out)` references into it, and the `init == true` build call
//!   sizes/seeds buffers from the build-time input values without running
//!   per-tick side effects (see [`op`] for the conventions). Because operators
//!   are plain segments, they compose with `flowgraph`'s combinators
//!   (`then`/`fork`/`par`) and the `flowgraph::segment!` fusion macro as-is.
//! * The engine's input-notification gating prunes the recompute cone: an
//!   [`Operator`](flowgraph::typed::Operator)'s compute path fires iff ≥1 input
//!   notified (else its `passthrough` re-emits the previous output, un-notified).
//! * **The contract that makes this sound: *no-notify ⟹ output unchanged*.**
//!   An operator that does not notify must leave its output value unchanged, so
//!   every consumer may treat a non-notifying input as its last notified value.
//!   That is what lets carry readers ([`Stack`] / [`StackView`]) read
//!   un-notified inputs, and what lets zero-copy view chains
//!   ([`Gate`] → [`SliceView`] → [`StackView`]) work without materializing every
//!   edge. See the [`op`] module conventions for the full statement; it is a
//!   producer-side duty obeyed by every operator here.
//! * Time is threaded out-of-band through a shared [`Clock`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//!
//! # Python operators (feature `python`)
//!
//! `PyOperator` runs a Python callable as a graph node, taking N `f64` array
//! inputs to one `f64` array output, with real NumPy. It runs on a single shared
//! embedded interpreter (GIL by default — NumPy/SciPy/solver work parallelizes
//! on the pool via GIL release; build against free-threaded CPython for
//! pure-Python parallelism too, no code change). Inputs are copied in
//! (NumPy-owned snapshots), output exposed zero-copy. Register via
//! [`Scenario::add_py_operator`](crate::Scenario::add_py_operator) (return mode)
//! / `add_py_operator_writing` (write mode) for lambdas, or
//! `add_py_operator_file` / `PyClassOperator` for class-based operators loaded
//! from `.py` files (see `flowops`). The [`python`] and [`pyhost`] submodule
//! docs cover the contracts, data model, and setup.

mod arith;
mod gating;
mod metrics;
mod num;
mod op;
#[cfg(feature = "python")]
mod pyhost;
#[cfg(feature = "python")]
mod python;
mod reshape;
mod rolling;
mod stocks;
mod structural;
mod traders;
mod transform;

#[cfg(feature = "python")]
pub use pyhost::{NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams};
#[cfg(feature = "python")]
pub use python::PyOperator;

pub use op::{ArrayInput, ArrayValue, Clock, StripNotify};
pub use gating::{Clocked, Count, Filter, Gate, Last, Record};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
};
pub use metrics::{AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Turnover, Volatility};
pub use reshape::{Concat, ConcatSync, Split, Stack, StackSync, StackSyncView, StackView};
pub use stocks::{Annualize, AnnualizeView, ForwardAdjust, ForwardAdjustViewDiv};
pub use transform::{Apply, ApplyInplace, Lag, Map, MapInplace, Select, SelectView, SliceView};
pub use arith::{
    Abs, Add, Ceil, Divide, Exp, Exp2, Floor, Log, Log2, Log10, Max, Min, Multiply, Negate, Pow,
    Recip, Round, Sign, Sqrt, Subtract,
};
pub use rolling::{
    Accumulator, CovarianceAccumulator, Ema, MeanAccumulator, Rolling, RollingCovariance,
    RollingMean, RollingSum, RollingVariance, SumAccumulator, VarianceAccumulator, Window,
};
pub use structural::{Cast, Id, Resample, Where};
pub use traders::{Benchmark, RandomTrader, SimpleTrader};

#[cfg(test)]
mod tests;
