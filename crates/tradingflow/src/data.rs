//! Core data types and trait machinery, re-exported from the
//! [`tradingflow_data`] crate.
//!
//! The primitive data containers that flow through the
//! [`operators`](crate::operators) engine live in their own crate so they
//! build and test against a minimal dependency set; this module is the facade
//! that keeps them reachable as `tradingflow::data::…` (and, through the crate
//! root, as `tradingflow::{Array, Series, Instant, …}`).
//!
//! # Sub-modules
//!
//! * [`array`](mod@array) — [`Array`] / [`ArrayView`]:
//!   dense rank-`N` array with row-major contiguous layout, and its borrowed
//!   strided view.
//! * [`series`] — [`Series`] / [`SeriesView`]:
//!   append-only time series of uniformly-shaped rank-`N` elements, and its
//!   borrowed window.
//! * [`time`] — [`Instant`] and [`Duration`]:
//!   SI-nanosecond timestamps anchored at the PTP epoch (1970-01-01 TAI).
//!
//! # This-module items
//!
//! * [`Scalar`] — marker trait for permitted array element types.

// The submodules are re-exported wholesale so path-based references
// (`crate::data::array::apply_unary`, `crate::data::array::Shape`) resolve
// unchanged.
pub use tradingflow_data::{array, series, time};

pub use tradingflow_data::{
    Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape,
    civil_from_days, days_from_civil, tai_to_utc, utc_to_tai,
};
