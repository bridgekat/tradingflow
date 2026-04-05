//! Built-in rolling (windowed) operators.
//!
//! Rolling operators take a [`Series<T>`](crate::Series) input and produce an
//! [`Array<T>`](crate::Array) output (`T: Scalar + Float`).  Each maintains
//! incremental state so that per-tick cost is O(1) per element (O(K²) for
//! the covariance matrix where K is the vector dimension).
//!
//! # Architecture
//!
//! The [`Accumulator`](accumulator::Accumulator) trait defines the incremental
//! `add` / `remove` / `write` interface.  The generic
//! [`Rolling<A>`](accumulator::Rolling) operator pairs an accumulator with a
//! [`Window`](accumulator::Window) strategy (count-based or time-delta-based)
//! and implements [`Operator`](crate::Operator).
//!
//! # Window strategies
//!
//! - **Count-based** ([`Rolling::count`](accumulator::Rolling::count)) — the
//!   window contains the last N elements.  Output is produced only once the
//!   window is full.
//! - **Time-delta-based** ([`Rolling::time_delta`](accumulator::Rolling::time_delta))
//!   — the window contains all elements within a time range of the most
//!   recent timestamp.  Output is produced as soon as at least one element is
//!   in the window.
//!
//! # Rolling operators
//!
//! - [`RollingSum`] — incremental sum via add/subtract.
//! - [`RollingMean`] — incremental mean via add/subtract.
//! - [`RollingVariance`] — population variance via `E[x²] − E[x]²`.
//! - [`RollingCovariance`] — pairwise covariance matrix of a 1D `[K]` input,
//!   producing a `[K, K]` output.
//!
//! # Other operators
//!
//! - [`Ema`] — exponential moving average with window-bounded weights (does
//!   not use the `Accumulator` abstraction).

pub mod accumulator;
pub mod covariance;
pub mod ema;
pub mod mean;
pub mod sum;
pub mod variance;

pub use accumulator::{Accumulator, Rolling, Window};
pub use covariance::CovarianceAccumulator;
pub use ema::Ema;
pub use mean::MeanAccumulator;
pub use sum::SumAccumulator;
pub use variance::VarianceAccumulator;

/// Element-wise rolling sum.
pub type RollingSum<T> = Rolling<SumAccumulator<T>>;

/// Element-wise rolling mean.
pub type RollingMean<T> = Rolling<MeanAccumulator<T>>;

/// Element-wise rolling population variance.
pub type RollingVariance<T> = Rolling<VarianceAccumulator<T>>;

/// Pairwise rolling covariance matrix (`[K] → [K, K]`).
pub type RollingCovariance<T> = Rolling<CovarianceAccumulator<T>>;
