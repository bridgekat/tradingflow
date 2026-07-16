#![doc = include_str!("../README.md")]

pub mod array;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView, Shape};
pub use series::{Retention, Series, SeriesView};
pub use time::{Duration, Instant, civil_from_days, days_from_civil, tai_to_utc, utc_to_tai};

/// A permitted array scalar type.
pub trait Scalar: Send + Sync + Clone + Default + 'static {}

impl<T: Send + Sync + Clone + Default + 'static> Scalar for T {}
