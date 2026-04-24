//! Re-exports of [`Source`] and [`ErasedSource`] from `crate::source`.
//!
//! The wavefront PoC reuses the existing source contract unchanged — the
//! tokio mpsc-channel lifecycle, `PeekableReceiver`, `PollFn` / `WriteFn`
//! machinery are identical.  Only the downstream consumption of events
//! (the graph flush) is rewritten.

pub use crate::source::{ErasedSource, InitFn, PollFn, Source, WriteFn};
