//! Core data types and trait machinery.
//!
//! This module groups the project's primitive data containers that flow
//! through the [`operators`](crate::operators) engine.
//!
//! # Sub-modules
//!
//! * [`array`] - [`Array`](array::Array): dense N-dimensional array with
//!   row-major contiguous layout.
//! * [`series`] - [`Series`](series::Series): append-only time series of
//!   uniformly-shaped arrays.
//! * [`time`] - [`Instant`](time::Instant) and [`Duration`](time::Duration):
//!   SI-nanosecond timestamps anchored at the PTP epoch (1970-01-01 TAI).
//!
//! # This-module items
//!
//! * [`Scalar`] - marker trait for permitted array element types.

pub mod array;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView, Shape};
pub use series::{Retention, Series};
pub use time::{Duration, Instant, tai_to_utc, utc_to_tai};

/// A permitted array scalar type.
pub trait Scalar: Sized + Send + Sync + Clone + Default + 'static {}

impl Scalar for () {}
impl Scalar for bool {}
impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for String {}
