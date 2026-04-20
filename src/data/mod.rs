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

pub mod array;
pub mod inputs;
pub mod series;
pub mod time;

pub use array::Array;
pub use inputs::{
    BitRead, FlatRead, FlatWrite, Input, InputTypes, SliceProduced, SliceRefs,
};
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

