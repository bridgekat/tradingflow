//! Built-in data sources for the DAG runtime.
//!
//! Every source in this module implements [`Source`](crate::Source) and is
//! registered into a [`Scenario`](crate::Scenario) via
//! [`Scenario::add_source`](crate::Scenario::add_source). Sources feed events
//! into the DAG through historical and live channels; the POCQ event loop
//! ([`Scenario::run`](crate::Scenario::run)) drains them in timestamp order.
//!
//! # Data sources
//!
//! - [`ArraySource`] — historical-only source backed by pre-loaded timestamp
//!   and value arrays. Each event carries an `Array<T>`. Requires a tokio
//!   runtime.
//! - [`CsvSource`] — historical-only source that reads a timestamp column and
//!   named value columns from a CSV file on disk. Outputs `Array<f64>`. Requires
//!   a tokio runtime.
//! - [`IterSource`] — source driven by an arbitrary `(timestamp, value)`
//!   iterator. More flexible than `ArraySource`; supports lazy/computed sequences
//!   and arbitrary output types. Requires a tokio runtime.
//!
//! # Clock sources
//!
//! Clock sources emit `()` events at specified timestamps and are used as
//! triggers for periodic operators.
//!
//! - [`clock`] — clock from explicit timestamps.
//! - [`daily_clock`] — daily timestamps (midnight in a given IANA timezone).
//! - [`monthly_clock`] — monthly timestamps (first of each month in a given
//!   IANA timezone).

pub mod array_source;
pub mod clock;
pub mod csv_source;
pub mod iter_source;

pub use array_source::ArraySource;
pub use clock::{clock, daily_clock, monthly_clock};
pub use csv_source::CsvSource;
pub use iter_source::IterSource;
