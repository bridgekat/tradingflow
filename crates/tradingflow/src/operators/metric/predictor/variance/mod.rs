//! Quality of predictions against a variance (second-moment) target.

#[cfg(feature = "python")]
mod log_likelihood;
#[cfg(feature = "python")]
mod minimum_variance;

#[cfg(feature = "python")]
pub use log_likelihood::log_likelihood;
#[cfg(feature = "python")]
pub use minimum_variance::minimum_variance;
