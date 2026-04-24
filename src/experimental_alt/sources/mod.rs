//! Re-exports of existing data sources for the wavefront runtime.
//!
//! Source implementations are reused unchanged — the `Source` trait
//! contract hasn't changed (mpsc channels + type-erased poll/write
//! functions), only the downstream consumer (the graph flush) has been
//! rewritten.

pub use crate::sources::ArraySource;
pub use crate::sources::clock;
