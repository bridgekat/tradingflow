//! Built-in rolling (windowed) operators.
//!
//! Every operator in this module implements [`Operator`](crate::Operator) with
//! [`Series<T>`](crate::Series) input and [`Array<T>`](crate::Array) output
//! (`T: Scalar + Float`). Each
//! maintains incremental state so that per-tick cost is O(1) per element
//! (O(K^2) for the covariance matrix where K is the vector dimension).
//!
//! All rolling operators share the following invariants:
//!
//! - **Window size**: set at construction (`window >= 1`). Before the window
//!   is full, computation uses all values seen so far.
//! - **NaN propagation**: if any value within the current window is NaN for a
//!   given element position, the output for that element is NaN. Once the NaN
//!   is evicted from the window, valid output resumes.
//! - **Element independence**: for vector inputs, NaN in one element position
//!   does not affect other positions.
//!
//! # Operators
//!
//! - [`RollingSum`] — incremental sum via add/subtract.
//! - [`RollingMean`] — incremental mean via add/subtract.
//! - [`RollingVariance`] — population variance via `E[x^2] - E[x]^2`.
//! - [`RollingCovariance`] — pairwise covariance matrix of a 1D `[K]` input,
//!   producing a `[K, K]` output. Uses incremental cross-product sums.
//! - [`Ema`] — exponential moving average with window-bounded weights.
//!   Constructors: [`Ema::new`] (explicit alpha), [`Ema::with_span`],
//!   [`Ema::with_half_life`].
//! - [`ForwardFill`] — replaces NaN with the last valid observation per
//!   element (no window parameter; stateful but not windowed).

pub mod covariance;
pub mod ema;
pub mod ffill;
pub mod mean;
pub mod sum;
pub mod variance;

pub use covariance::RollingCovariance;
pub use ema::Ema;
pub use ffill::ForwardFill;
pub use mean::RollingMean;
pub use sum::RollingSum;
pub use variance::RollingVariance;
