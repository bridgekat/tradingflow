//! Quality of predictions against a mean (first-moment) target.

#[cfg(feature = "python")]
mod information_coefficient;

#[cfg(feature = "python")]
pub use information_coefficient::information_coefficient;
