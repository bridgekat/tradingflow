//! Portfolio construction operators.
//!
//! # Inputs
//!
//! - `rebalance_signal`: a scalar signal indicating a rebalance.
//! - `universe`: an `[N]` vector, positive on the tradable stocks and read as
//!   benchmark weights by the portfolios that need one.
//! - `predict`: the predicted moments — an `[N]` return vector, an `[N, N]`
//!   covariance, or both, depending on the portfolio.
//!
//! # Outputs
//!
//! - `signal`: fires once per rebalance.
//! - `weights`: an `[N]` book, recomputed at each rebalance and retained in
//!   between. Stocks outside the universe, and stocks whose predicted moments
//!   are non-finite, hold zero.
//!
//! The moments are read as *whatever stands on the wire*, not as an event, so
//! a portfolio rebalancing faster than its predictor simply reuses the last
//! prediction.
//!
//! # Log versus linear returns
//!
//! Predictors work in log returns, since those aggregate additively over time.
//! Optimizers work in linear returns, since those aggregate additively across
//! a portfolio — a weighted sum of log returns is not the log return of the
//! weighted sum, and optimizing as though it were quietly misprices every
//! book. [`Config::logarithmic`] puts the lognormal moment map between them
//! and is on by default; turn it off only when the input is already linear, or
//! is not a return at all.

#[cfg(feature = "python")]
mod config;

#[cfg(feature = "python")]
pub mod mean;
#[cfg(feature = "python")]
pub mod mean_variance;
#[cfg(feature = "python")]
pub mod variance;

#[cfg(feature = "python")]
pub use config::Config;
