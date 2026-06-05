//! `flow` — the new synchronous, parallel engine layer, built on `flowgraph`.
//!
//! Migration scaffolding (branch `flowgraph-rebase`). This module coexists with
//! the legacy async [`scenario`](crate::scenario) engine so operators can be
//! ported and **differential-tested** batch by batch before cutover, rather
//! than in one big-bang rewrite.
//!
//! # Design (validated end-to-end before landing here)
//!
//! * Operators implement [`Operator`] — the TradingFlow operator contract
//!   (`compute -> bool`, a single typed output) expressed over `flowgraph`'s
//!   [`Ports`](flowgraph::typed::Ports) so the input / notify trees are the
//!   engine's own types. Operators are *pure* (no threaded timestamp); the few
//!   that stamp event time hold the [`Clock`] in their own state (e.g.
//!   [`Record`]). `State` and `Output` are `Send + Sync` (project decision: no
//!   operator instance is `!Sync`).
//! * [`Adapt`] bridges any [`Operator`] onto [`flowgraph::typed::Operator`],
//!   mapping the `bool` return directly onto the single output's notify flag.
//!   Combined with the engine's input-notification gating, this reproduces the
//!   legacy `did_produce` cone-prune exactly: a node fires iff ≥1 input
//!   notified, which equals the old "iff ≥1 input produced".
//! * Time is threaded out-of-band through a shared [`Clock`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//!
//! # Python operators (feature `pyflow`)
//!
//! `PyOperator` runs a Python callable as a graph node, taking N `f64` array
//! inputs to one `f64` array output, with real NumPy and true parallelism on a
//! free-threaded interpreter. Inputs are copied in (NumPy-owned snapshots) and
//! the output is exposed zero-copy. Register via `Scenario::add_py_operator`
//! (return mode) or `Scenario::add_py_operator_writing` (write mode). The
//! `python` submodule docs cover the contracts, the copy-in/zero-copy data
//! model, the retention/safety contract, and the free-threaded build/run setup.
//!
//! # Status
//!
//! All legacy operators ported with per-operator differential tests vs the old
//! engine; the async source/event-loop driver and the Python-operator bridge are
//! landed. Remaining before cutover: concurrency hardening (TSan/loom on the
//! notify/counter protocol) and an end-to-end parity harness, then removal of the
//! legacy `operator`/`operators`/`scenario`/`bridge` modules.

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
pub use pyhost::{NativeArrayView, NativeSeriesView, PyClassOperator, PyParams};
#[cfg(feature = "pyflow")]
pub use python::PyOperator;

pub use op::{Adapt, Clock, Operator};
pub use session::{Scenario, Session, ShutdownFlag};
pub use ops::{Clocked, Const, Count, Filter, Last, Record};
pub use num::{
    Clamp, Diff, Fillna, ForwardFill, Gaussianize, PctChange, Percentile, Standardize, Winsorize,
};
pub use metrics::{AverageReturn, CompoundReturn, Drawdown, SharpeRatio, Volatility};
pub use reshape::{Concat, ConcatSync, Stack, StackSync};
pub use stocks::{Annualize, ForwardAdjust};
pub use transform::{Apply, ApplyInplace, Lag, Map, MapInplace, Select};
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
