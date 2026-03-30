//! Miscellaneous utilities.
//!
//! # Public API
//!
//! - [`Schema`] — bidirectional name-to-position mapping for labelling array
//!   axes. Provides `O(1)` lookup in both directions, plus helpers for
//!   sub-selection ([`Schema::select`]) and concatenation
//!   ([`Schema::concat`]). This is a construction-time helper and is not
//!   embedded in the dataflow graph or carried by arrays at runtime.

pub mod schema;

pub use schema::Schema;
