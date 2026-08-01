//! Python segment wrapper.
//!
//! # Memory ownership model
//!
//! ## Inputs
//!
//! Inputs to Python segments are allowed to reference Rust-owned memory
//! through [`NativeView`], which implement the PEP 3118 buffer protocol.
//!
//! However, Python segments must not keep them in node state, since the memory
//! is only guaranteed to have input lifetime which the node state outlives
//! (see [`Segment`](crate::graph::Segment) signature). To prevent this, after
//! each call to some Python function taking [`NativeView`] as arguments,
//! the Rust host must check that the views' export counters are zero through
//! [`NativeView::invalidate`] and panic if not. [`Scope::close`] checks for
//! this condition automatically; see below.
//!
//! ## Outputs
//!
//! Outputs from Python segments are passed as NumPy arrays, and are always
//! copied into Rust-owned memory before dropped.
//!
//! This simplifies the ownership model: after a Python segment returns, there
//! should be no references into Python-owned memory in Rust and vice versa.
//! It also makes export-count-checking easy for inputs: if no copy is
//! performed, the output array can hold references into the inputs, which
//! can make the export counts nonzero and trigger an undesired panic.
//!
//! > Although Rust segments are allowed to return references into their inputs,
//! > supporting this in Python segments would require more complex machinery
//! > that whitelists input export counts if they only come from outputs -
//! > difficult to do reliably.
//!
//! # The native view
//!
//! [`NativeView`] is a deliberately minimal `#[pyclass]`, which is only used
//! through wrapper classes in `python/tradingflow/views.py`. It exposes the
//! core PEP 3118 buffer protocol to Python code so they can access Rust-owned
//! memory, and implements checked invalidation in [`NativeView::invalidate`].
//!
//! [`NativeView`] should be safe to use even with free-threaded Python
//! interpreters.
//!
//! # The scope helper
//!
//! [`Scope`] can be used to create tracked [`NativeView`]s for one Python
//! function call, via the [`Scope::array`] and [`Scope::series`] methods.
//!
//! After a python function returns, its return values are copied to Rust-owned
//! memory and dropped. After that, [`Scope::close`] can be used to check every
//! tracked export counter and invalidate every tracked view, returning [`Err`]
//! if any exports are still outstanding. Panicking on [`Err`] is enough to
//! prevent safety violations:
//!
//! - A **captured pointer** (the operator stored `np.asarray(view)` or slices
//!   of it) keeps an export outstanding. The pointer cannot be revoked, so the
//!   escape is reported and the host panics.
//! - A **captured [`NativeView`]** that was never exported is invalidated,
//!   making its data pointer no longer accessible: later use raises
//!   `BufferError` instead of reading from a dangling pointer.
//!
//! What no refcount or counter scheme can see is a laundered address
//! (`arr.ctypes.data` stashed as an integer, or a C extension caching the
//! pointer). That is outside the contract — the check defends against
//! accidents, not against deliberate circumvention.
//!
//! # Why binding a view is `unsafe`
//!
//! Binding erases a borrow: [`Scope::array`], [`Scope::series`] and the
//! [`NativeView`] constructors under them take a view with a lifetime and hand
//! back a Python object holding a bare pointer. Nothing left in the type system
//! tracks when that memory dies, and no lifetime discipline could — the escape
//! route is Python's own object graph, which Rust cannot see into. That is
//! precisely why the check is dynamic, and why the obligation to run it has to
//! be carried by the caller. So they are `unsafe fn`: the caller must close the
//! scope while the payload is still alive, and must treat [`Err`] as fatal.
//!
//! Dropping a scope instead of closing it is *not* enough. It invalidates every
//! view, so nothing can acquire the payload afterwards, but it skips the export
//! check — a pointer Python captured during the call then goes unreported and
//! dangles once the payload dies. Detection is the last line of defence here,
//! and it only defends if it actually runs.

mod scalar;
mod scope;
mod view;

pub use scalar::{DType, NativeScalar};
pub use scope::{EscapedViewsError, Scope};
pub use view::NativeView;
