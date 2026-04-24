//! Wavefront execution model — proof of concept.
//!
//! A re-imagining of the TradingFlow runtime where the 2D grid of
//! `(node, tick)` pairs is scheduled as a wavefront: a cell is ready
//! when all its upstream inputs at the same tick have completed
//! (horizontal dependency), AND the same node at the previous tick has
//! completed if the operator is stateful (vertical dependency).
//!
//! # Architecture
//!
//! * [`Operator`] — replacement trait adding `is_stateful()` and
//!   `State: Clone`.
//! * [`Source`] — batch source trait (all events known upfront).
//! * [`WavefrontScenario`] — graph builder and runner.
//!
//! # Sub-modules
//!
//! * [`operators`] — PoC operator implementations.
//! * [`adapter`] — bridges `crate::Operator` impls to
//!   [`experimental::Operator`].

pub mod adapter;
pub mod graph;
pub mod operator;
pub mod operators;
pub mod scheduler;
pub mod scenario;
pub mod source;
pub mod storage;

pub use operator::Operator;
pub use scenario::WavefrontScenario;
pub use source::Source;
pub use storage::VersionedRing;
