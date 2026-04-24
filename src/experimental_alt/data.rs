//! Re-exports of core data types from `crate::data`.
//!
//! The wavefront PoC reuses the existing [`Array`], [`Series`], `Input`
//! marker, `InputTypes` machinery, cursors, and scalar types verbatim.
//! Only the runtime (graph, scheduler, output storage) is rewritten.

pub use crate::data::{
    Array, BitRead, Duration, FlatRead, FlatWrite, Input, InputTypes, Instant, PeekableReceiver,
    Scalar, Series, SliceProduced, SliceRefs, tai_to_utc, utc_to_tai,
};
