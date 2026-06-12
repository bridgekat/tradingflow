//! `flow` — the new synchronous, parallel engine layer, built on `flowgraph`.
//!
//! Migration scaffolding (branch `flowgraph-rebase`). This module coexists with
//! the legacy async [`scenario`](crate::scenario) engine so operators can be
//! ported and **differential-tested** batch by batch before cutover, rather
//! than in one big-bang rewrite.
//!
//! # Design (validated end-to-end before landing here)
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
//! * The engine's input-notification gating reproduces the legacy
//!   `did_produce` cone-prune exactly: an [`Operator`]'s compute path fires iff
//!   ≥1 input notified (else its `passthrough` re-emits the previous output,
//!   un-notified), which equals the old "iff ≥1 input produced".
//! * Source cells are `push_source` nodes poked through the typed
//!   `Graph::state_mut(SourceHandle<T>)` (which marks the dirty cone); the
//!   async [`Source`](crate::source::Source) feed is driven by [`Session`].
//! * Time is threaded out-of-band through a shared [`Clock`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//!
//! # Python operators (feature `pyflow`)
//!
//! `PyOperator` runs a Python callable as a graph node, taking N `f64` array
//! inputs to one `f64` array output, with real NumPy. It runs on a single shared
//! embedded interpreter (GIL by default — NumPy/SciPy/solver work parallelizes
//! on the pool via GIL release; build against free-threaded CPython for
//! pure-Python parallelism too, no code change). Inputs are copied in
//! (NumPy-owned snapshots), output exposed zero-copy. Register via
//! `Scenario::add_py_operator` (return mode) / `add_py_operator_writing` (write
//! mode) for lambdas, or `add_py_operator_file` / `PyClassOperator` for
//! class-based operators loaded from `.py` files (see `flowops`). The `python`
//! and `pyhost` submodule docs cover the contracts, data model, and setup.

mod arith;
mod metrics;
mod num;
mod op;
mod ops;
#[cfg(feature = "pyflow")]
mod pyhost;
#[cfg(feature = "pyflow")]
mod python;
mod reshape;
mod rolling;
mod session;
mod stocks;
mod structural;
mod transform;

#[cfg(feature = "pyflow")]
pub use pyhost::{NativeArrayView, NativeSeriesView, PyArgs, PyClassOperator, PyParams};
#[cfg(feature = "pyflow")]
pub use python::PyOperator;

// The `flowgraph` vocabulary the flow layer is written in, re-exported for
// downstream graph-building code. (`flowgraph::typed::Id` is NOT re-exported —
// `flow::Id` is the structural identity operator below; reach the combinator
// via its full path.)
pub use flowgraph::typed::{
    Arena, Handle, Interface, InterfaceHandles, Operator, Port, RefPort, RefPorts, RefSource,
    RefViewPort, RefViewPorts, Scalar as ScalarValue, Segment, SegmentExt, Source as ValueSource,
    SourceHandle, ValueView, ViewPort, ViewSource,
};

pub use op::{ArrayInput, ArrayValue, Clock, StripNotify};
pub use session::{Scenario, Session, ShutdownFlag};
pub use ops::{Clocked, Count, Filter, Gate, Last, Record};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
};
pub use metrics::{AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Volatility};
pub use reshape::{Concat, ConcatSync, Split, Stack, StackSync};
pub use stocks::{Annualize, AnnualizeView, ForwardAdjust, ForwardAdjustViewDiv};
pub use transform::{Apply, ApplyInplace, Lag, Map, MapInplace, Select, SelectView};
pub use arith::{
    Abs, Add, Ceil, Divide, Exp, Exp2, Floor, Log, Log2, Log10, Max, Min, Multiply, Negate, Pow,
    Recip, Round, Sign, Sqrt, Subtract,
};
pub use rolling::{
    Accumulator, CovarianceAccumulator, Ema, MeanAccumulator, Rolling, RollingCovariance,
    RollingMean, RollingSum, RollingVariance, SumAccumulator, VarianceAccumulator, Window,
};
pub use structural::{Cast, Id, Resample, Where};

#[cfg(test)]
mod tests;
