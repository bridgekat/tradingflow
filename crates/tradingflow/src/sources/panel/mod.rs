//! Panel sources from file or database tables.
//!
//! # Table format
//!
//! Source in this module expect *long-format* tables, which should contain the
//! following columns:
//!
//! - One timestamp (datetime) column, sorted in ascending order.
//! - `N >= 0` array-index columns, which contain either named (string) or
//!   numeric (integer) labels that indexes into an `N`-dimensional array.
//! - `M >= 0` array-value columns, each contain scalar values to be filled
//!   into an `N`-dimensional array.
//!
//! Each group of rows sharing the same timestamp is a single *cross-section*,
//! which will be emitted as a single `N`-dimensional array at that timestamp.
//! In each cross-section, indices which do not appear in any of the rows are
//! considered *inactive*, whose array cells preserve the last emitted value.
//! The source also emits a signal array marking the *active* cells at each
//! timestamp.
//!
//! `N = 0` can be used for scalar value streams (table does not need to
//! contain any index columns), and `M = 0` can be used for signal-only streams
//! (table does not need to contain any value columns). Therefore, a table
//! containing only a timestamp column is valid as a single boolean signal
//! which is `true` only at the timestamps present in the table.
//!
//! ## Timestamp columns
//!
//! The timestamp column is specified by its name in the file or database table.
//!
//! It should have a naive timestamp type convertible to
//! [`data::Instant`](crate::data::Instant), and an out-of-range value panics
//! at runtime. Timestamps must be sorted in ascending order so that the table
//! can be streamed; if not, the table must be pre-processed before being used
//! as a source.
//!
//! ## Array-index columns
//!
//! The number `N` of array-index columns must be statically specified, so that
//! the output array dimensions are statically known.
//!
//! Each array-index column is specified by a pair `(name, axis)`, where `name`
//! is the name of the column in the file or database table, and `axis` is a
//! [`data::Axis`](crate::data::Axis) which defines the mapping from the
//! column's values to array indices. An unexpected value in the column
//! (not in the axis range) panics at runtime.
//!
//! ## Array-value columns
//!
//! The number `M` of array-value columns can be dynamically specified, but
//! each column must map to the same scalar type (since a dynamic-length group
//! of ports must be uniformly typed).
//!
//! Each array-value column is specified by its name in the file or database
//! table. An unexpected value in the column (not of the specified scalar type)
//! panics at runtime.
//!
//! Support for static `M` and heterogeneous value types is planned for the
//! future.
//!
//! # Outputs
//!
//! - `signals`: the signal array marking active cells.
//! - `values`: `M` value arrays, each containing the values of a value column.

mod base;
mod csv;
mod parquet;

pub use base::{Panel, PanelBatch, PanelState, Reader};
pub use csv::csv;
pub use parquet::parquet;
