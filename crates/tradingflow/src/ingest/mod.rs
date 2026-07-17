//! The event loop driver.
//!
//! # Constructing scenarios
//!
//! A [`Scenario`] is used to construct a [`Session`].
//!
//! - [`Scenario::source`] adds a data source to the graph, returning its
//!   output port handle.
//! - [`Scenario::segment`] adds a segment (sub-graph) to the graph, returning
//!   its output port handles.
//! - [`Scenario::build`] finalizes into a [`Session`].
//!
//! A [`Session`] represents a complete strategy.
//!
//! - [`Session::step`] ingests and processes a single batch of events.
//! - [`Session::run`] runs batches repeatedly until all sources are exhausted.
//! - [`Session::view`] reads values on wires. This can only be used in-between
//!   batches.

mod clock;
mod feed;
mod queue;
mod scenario;
mod source;

pub use clock::{Clock, WallClock};
pub use feed::{Event, Feed, LazyFeed, Stamp, StreamFeed};
pub use queue::Queue;
pub use scenario::{Scenario, Session};
pub use source::EventSource;
