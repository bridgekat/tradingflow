//! Generic N-dimensional arrays and time series — the primitive data
//! containers that flow through the TradingFlow operator engine.
//!
//! This crate is the data model, factored out so it builds and tests against
//! its own minimal dependency set (std, plus `hifitime` for the calendar
//! arithmetic), independent of the graph engine and the operator library that
//! layer on top. The `tradingflow` crate re-exports everything here through its
//! `data` module, so strategy code still reaches these types through
//! `tradingflow` alone.
//!
//! # Sub-modules
//!
//! * [`array`] — [`Array`] / [`ArrayView`]:
//!   dense rank-`N` array with row-major contiguous layout, and its borrowed
//!   strided view.
//! * [`series`] — [`Series`] / [`SeriesView`]:
//!   append-only time series of uniformly-shaped rank-`N` elements, and its
//!   borrowed window. A series is stamped with [`Instant`] timestamps, which is
//!   why the time types live in this crate too.
//! * [`time`] — [`Instant`] and [`Duration`]:
//!   SI-nanosecond timestamps anchored at the PTP epoch (1970-01-01 TAI).
//!
//! # This-crate items
//!
//! * [`Scalar`] — marker trait for permitted array element types.

pub mod array;
pub mod series;
pub mod time;

pub use array::{Array, ArrayView, Shape};
pub use series::{Retention, Series, SeriesView};
pub use time::{Duration, Instant, civil_from_days, days_from_civil, tai_to_utc, utc_to_tai};

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
