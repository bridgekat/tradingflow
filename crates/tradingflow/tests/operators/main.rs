//! Integration tests for the built-in operators.
//!
//! One module per operator module, plus [`compose`], which covers what no
//! single-operator test can: `segment!` fusion, hoisted-versus-self-recording
//! equivalence, and parallel execution.
//!
//! Every test drives a real [`Graph`](tradingflow::graph::typed::Graph): build
//! a wiring, poke sources through `state_mut`, `stabilize`, then assert on the
//! views. Shared constructors, assertions and probe segments live in
//! [`harness`].

mod harness;

mod array;
mod compose;
mod elem;
mod feature;
mod metric;
mod portfolio;
mod predictor;
mod rolling;
mod series;
mod signal;
mod stats;
mod trader;
