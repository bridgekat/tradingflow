//! Core data types and trait machinery.
//!
//! This module groups the project's primitive data containers and the
//! glue traits that describe how they flow through a [`Scenario`](crate::Scenario).
//!
//! # Sub-modules
//!
//! * [`array`] — [`Array`](array::Array): dense N-dimensional array with
//!   row-major contiguous layout.
//! * [`series`] — [`Series`](series::Series): append-only time series of
//!   uniformly-shaped arrays.
//! * [`time`] — [`Instant`](time::Instant) and [`Duration`](time::Duration):
//!   SI-nanosecond timestamps anchored at the PTP epoch (1970-01-01 TAI).
//! * [`inputs`] — [`InputTypes`](inputs::InputTypes) and friends: recursive
//!   description of operator inputs, cursor-style readers/writers, and the
//!   leaf ([`Input<T>`](inputs::Input)) / slice ([`Slice<T>`](inputs::Slice))
//!   wrappers.
//!
//! # This-module items
//!
//! * [`Scalar`] — marker trait for permitted array element types.
//! * [`PeekableReceiver`] — tokio mpsc wrapper with one-slot peek buffer,
//!   used by sources.
//! * [`Notify`] — notification context passed to
//!   [`Operator::compute`](crate::Operator::compute).

pub mod array;
pub mod inputs;
pub mod series;
pub mod time;

pub use array::Array;
pub use inputs::{FlatRead, FlatWrite, Input, InputTypes, SliceProduced, SliceRefs};
pub use series::Series;
pub use time::{Duration, Instant, tai_to_utc, utc_to_tai};

use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// A permitted array scalar type.
pub trait Scalar: Sized + Send + Sync + Clone + Default + 'static {}

impl Scalar for () {}
impl Scalar for bool {}
impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}
impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for String {}

/// Peekable wrapper around Tokio [`mpsc::Receiver`] with a one-slot pending
/// buffer.
///
/// Supports a two-phase peek-then-consume protocol:
/// [`poll_pending`](Self::poll_pending) peeks the next item without consuming
/// it, and [`take_pending`](Self::take_pending) later extracts the buffered
/// item.
#[derive(Debug)]
pub struct PeekableReceiver<T: Send + 'static> {
    rx: mpsc::Receiver<T>,
    pending: Option<T>,
}

impl<T: Send + 'static> PeekableReceiver<T> {
    /// Create a new peekable receiver wrapping the given channel.
    pub fn new(rx: mpsc::Receiver<T>) -> Self {
        Self { rx, pending: None }
    }

    /// Poll for the next item without consuming it.
    ///
    /// If an item is already buffered, returns a reference to it immediately.
    /// Otherwise polls the underlying receiver, buffering any received item
    /// and returning a reference to it.
    pub fn poll_pending(&mut self, cx: &mut Context<'_>) -> Poll<Option<&T>> {
        if self.pending.is_none() {
            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(item)) => self.pending = Some(item),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(self.pending.as_ref())
    }

    /// Take the buffered item, if any.
    pub fn take_pending(&mut self) -> Option<T> {
        self.pending.take()
    }
}

/// Notification context for [`Operator::compute`](crate::Operator::compute).
///
/// Provides two views of which inputs produced new output in the current
/// flush cycle:
///
/// * [`produced`](Self::produced) — zero-alloc lazy iterator over **local**
///   input positions (0-based relative to this operator) that produced.
/// * [`input_produced`](Self::input_produced) — `Box<[bool]>` indexed by
///   local position (O(1) per-position check).  Allocates on first call.
///
/// # Offset and operator transformers
///
/// [`Notify`] carries an `offset` that maps global flat positions (as stored
/// in the graph) to local positions seen by the operator.  For all top-level
/// operators `offset = 0` — there is no overhead.  Operator transformers
/// such as [`Clocked<O>`](crate::operators::Clocked) use
/// [`skip_leading`](Self::skip_leading) to create an inner `Notify` with
/// an incremented offset so the wrapped operator sees correctly remapped
/// positions — with zero allocation.
///
/// Operators that never call `produced()` or `input_produced()` pay zero
/// cost regardless of the offset value.
pub struct Notify<'a> {
    incoming: &'a [usize],
    num_inputs: usize,
    /// Maps global flat position → local position: local = global − offset.
    /// Zero for all top-level operators; non-zero only inside transformers.
    offset: usize,
}

impl<'a> Notify<'a> {
    /// Create a top-level notification context (zero-cost, offset = 0).
    ///
    /// `incoming` lists the flat global input positions that produced.
    /// `num_inputs` is the total number of inputs for the operator.
    #[inline(always)]
    pub fn new(incoming: &'a [usize], num_inputs: usize) -> Self {
        Notify { incoming, num_inputs, offset: 0 }
    }

    /// Zero-allocation lazy iterator over **local** positions that produced.
    ///
    /// Yields positions in `0..num_inputs`, filtering and remapping the raw
    /// global positions using the internal offset.  For top-level operators
    /// (offset = 0) this is equivalent to iterating the raw slice.
    ///
    /// Prefer this over [`input_produced`](Self::input_produced) on hot paths
    /// that only need to check a small number of positions.
    #[inline(always)]
    pub fn produced(&self) -> impl Iterator<Item = usize> + '_ {
        let (off, n) = (self.offset, self.num_inputs);
        self.incoming.iter().filter_map(move |&p| {
            (p >= off && p - off < n).then_some(p - off)
        })
    }

    /// Raw global positions slice — for the Python bridge only.
    ///
    /// Returns the unshifted flat positions stored in the graph.  Prefer
    /// [`produced`](Self::produced) for operator logic.
    #[inline(always)]
    pub fn produced_raw(&self) -> &[usize] {
        self.incoming
    }

    /// Returns a boolean slice indexed by **local** position: `true` if
    /// that input produced this cycle.
    ///
    /// Allocates on each call.  For hot paths, prefer
    /// [`produced`](Self::produced).
    pub fn input_produced(&self) -> Box<[bool]> {
        let mut flags = vec![false; self.num_inputs].into_boxed_slice();
        for p in self.produced() {
            flags[p] = true;
        }
        flags
    }

    /// Create an inner [`Notify`] for a wrapped operator, skipping the
    /// first `n` inputs of the current operator.
    ///
    /// Used by operator transformers (e.g.
    /// [`Clocked<O>`](crate::operators::Clocked)) to pass a correctly
    /// remapped notification to the inner operator without allocating.
    ///
    /// The inner `Notify` covers positions `n..num_inputs` of the current
    /// operator, re-mapped to `0..(num_inputs - n)`.
    #[inline(always)]
    pub fn skip_leading(&self, n: usize) -> Notify<'a> {
        Notify {
            incoming: self.incoming,
            num_inputs: self.num_inputs.saturating_sub(n),
            offset: self.offset + n,
        }
    }

    /// Raw pointer to the incoming positions slice (for Python bridge).
    #[inline(always)]
    pub fn incoming_ptr(&self) -> *const usize {
        self.incoming.as_ptr()
    }

    /// Length of the incoming positions slice (for Python bridge).
    #[inline(always)]
    pub fn incoming_len(&self) -> usize {
        self.incoming.len()
    }

    /// Total number of inputs (for Python bridge).
    #[inline(always)]
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }
}
