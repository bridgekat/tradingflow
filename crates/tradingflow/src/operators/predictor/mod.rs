//! Prediction model operators.
//!
//! # Inputs
//!
//! - `sample_signal`: a scalar signal indicating a sampling period.
//! - `features`: an `[N, F]` panel of cross-sectional features for that
//!   period.
//! - `target`: an `[N]` vector of realized targets over that period.
//! - `rebalance_signal`: a scalar signal indicating a prediction period.
//! - `universe`: an `[N]` vector, positive on the stocks to predict for.
//!
//! # Outputs
//!
//! - `signal`: fires once per rebalance.
//! - `prediction`: the predicted cross-section, recomputed at each rebalance
//!   and retained in between. Stocks the model could not price — outside the
//!   universe, short of `min_periods` coverage, or carrying non-finite
//!   features — are `NaN`.
//!
//! A predictor fits on a window of past `(features, target)` pairs, sampled on
//! `sample_signal` and paired with a forward offset: `features[i]` predicts
//! `target[i + target_offset]`. Sampling is typically daily while rebalancing
//! is much less frequent, so the two cadences are separate signals; wiring the
//! same signal to both is fine and means "refit every sample".
//!
//! # Recording history
//!
//! Every input is the *latest* cross-section, so the window is recorded on the
//! Python side rather than read back from an upstream
//! [`record_all`](crate::operators::series::record_all). [`Config::max_periods`]
//! sizes it, and leaving it `None` retains every pair — unbounded memory, so
//! prefer a window on long runs.
//!
//! The incremental predictors are the exception: they fold each pair into a
//! sufficient-statistic pool as soon as its forward target lands, so they hold
//! only `target_offset + 1` cross-sections whatever the window, and their
//! refit cost is independent of how far back history reaches.

#[cfg(feature = "python")]
mod config;

#[cfg(feature = "python")]
pub mod mean;
#[cfg(feature = "python")]
pub mod variance;

#[cfg(feature = "python")]
pub use config::Config;
