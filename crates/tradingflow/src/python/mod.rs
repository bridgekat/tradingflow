//! Python segment wrapper.
//!
//! # Memory ownership model
//!
//! Everything crossing the FFI boundary is **copied**. No pointer into
//! Rust-owned memory ever reaches Python, and no pointer into Python-owned
//! memory survives past the call that produced it — after a Python segment
//! returns, there are no references across the boundary in either direction.
//!
//! ## Inputs
//!
//! [`to_numpy`] copies payload views into fresh **Python-owned** NumPy
//! arrays. Because the arrays own their data, Python
//! segments may freely store, mutate, slice or share them — including keeping
//! them in node state across generations, the thing a borrowed view could
//! never allow (node state outlives the input lifetime; see the
//! [`Segment`](crate::graph::Segment) signature). There is no contract for
//! operator code to uphold and no host-side check to run.
//!
//! ## Outputs
//!
//! [`from_numpy`] copies the value a Python segment returns into a Rust-owned
//! buffer before the value is dropped. The value is normalized with
//! `numpy.ascontiguousarray` on the way (so a list or a scalar is accepted,
//! and returning an input array unchanged is fine), and the boundary
//! negotiates by element count, not shape.
//!
//! ## Costs, and the road not taken
//!
//! Rust segments can return references into their inputs; copies mean Python
//! segments cannot. That property is cheap in Rust because the borrow checker
//! polices it at compile time. A zero-copy bridge (PEP 3118 views with export
//! counting) was built and retired here — see git history — because the
//! equivalent policing at the FFI boundary has to happen at runtime, its only
//! sound failure mode is aborting the process (a captured pointer cannot be
//! revoked, and a Python thread may already be using it), and the contract
//! ("never retain a view or anything derived from it") lands on operator
//! authors who NumPy has trained to do exactly the opposite.
//!
//! The copies this model pays for are one input copy and one output copy per
//! `compute`, which is negligible against the NumPy/SciPy/solver work Python
//! operators exist to run — in the strategy examples the solvers outweigh them
//! by orders of magnitude. What can grow is not the copying but the *history*:
//! since only the latest cross-section crosses, an operator that fits on a
//! window has to record that window itself. That cost is the operator's to
//! bound, and a model which does not read a panel should not retain one — see
//! `retain_features` in `tradingflow.predictor`.
//!
//! # What crosses the boundary
//!
//! **Arrays only, for now**: array and signal ports, their variadic groups,
//! and tuples thereof — the shapes [`PyInterface`] is implemented for.
//! Series stay on the Rust side; window them into arrays or reduce them with
//! native operators before wiring into a Python segment.
//!
//! # Python segments
//!
//! [`PySegment<I, O>`] is a graph node hosting a Python segment whose contract
//! mirrors the [`Segment`](crate::graph::Segment) trait method-for-method —
//! `init(inputs)`, `reset(inputs, state)`, `compute(inputs, state, instant)`
//! — with [`PyInterface`] packing the conversions on both sides: inputs
//! arrive as a flat tuple of owned NumPy arrays (one per leaf, in tree
//! order), and the returned outputs are copied into the node's Rust-owned
//! buffers, which back the views the node hands the engine. See
//! [`PySegment`] for the Python-side contract.

mod convert;
mod interface;
mod segment;

pub use convert::{from_numpy, from_numpy_into, to_numpy};
pub use interface::{GroupBuffers, PyInterface};
pub use segment::{
    PySegment, PyState, py_params, py_segment_file, py_segment_module, py_segment_source,
};
