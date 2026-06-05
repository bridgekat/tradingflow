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
//!   (`compute -> bool`, a threaded [`Instant`](crate::Instant), a single typed
//!   output) but expressed over `flowgraph`'s [`Ports`](flowgraph::typed::Ports)
//!   so the input / notify trees are the engine's own types. `State` and
//!   `Output` are `Send + Sync` (project decision: no operator instance is
//!   `!Sync`).
//! * [`Adapt`] bridges any [`Operator`] onto [`flowgraph::typed::Operator`],
//!   mapping the `bool` return directly onto the single output's notify flag.
//!   Combined with the engine's input-notification gating, this reproduces the
//!   legacy `did_produce` cone-prune exactly: a node fires iff ≥1 input
//!   notified, which equals the old "iff ≥1 input produced".
//! * Time is threaded out-of-band through a shared [`Clock`] the driver advances
//!   before each `stabilize` (only operators that stamp event time read it).
//!
//! # Status
//!
//! First operator batch + differential tests against the legacy engine's known
//! outputs, plus a parallel-execution gate (`Pool::new(N>0)`). The async
//! source/event-loop driver, the Python bridge, and the remaining operators are
//! follow-on increments.

mod arith;
mod metrics;
mod num;
mod op;
mod ops;
mod reshape;
mod rolling;
mod session;
mod stocks;
mod structural;
mod transform;

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
