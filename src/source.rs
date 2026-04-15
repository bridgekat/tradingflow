use std::any::TypeId;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

use crate::PeekableReceiver;
use crate::time::Instant;

/// An asynchronous data source that receives events via channels and writes
/// a typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) consumes the spec, producing two channel receivers
///    (historical + live) and the initial [`Output`](Self::Output).
/// 2. [`write`](Self::write) is called for each received event to update the
///    output.
pub trait Source: 'static {
    /// Channel event type.
    type Event: Send + 'static;
    /// Output type.
    type Output: Send + 'static;

    /// Consume the spec and produce channel receivers and initial output.
    #[allow(clippy::type_complexity)]
    fn init(
        self,
        timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, Self::Event)>,
        mpsc::Receiver<(Instant, Self::Event)>,
        Self::Output,
    );

    /// Update the output with a received event.
    ///
    /// Returns `true` if downstream propagation should occur.
    fn write(event: Self::Event, output: &mut Self::Output, timestamp: Instant) -> bool;

    /// Estimated total number of events this source will emit over its
    /// lifetime.  `None` for live / unbounded sources.
    ///
    /// Used only for progress reporting via
    /// [`Scenario::run`](crate::Scenario::run)'s `on_flush` callback —
    /// treated as advisory.  The default returns `None`.
    fn estimated_event_count(&self) -> Option<usize> {
        None
    }
}

/// Type-erased initialization closure for a source.
///
/// # Parameters
///
/// * `timestamp: Instant` — initial timestamp.
///
/// # Returns
///
/// * `hist_rx_ptr: *mut u8` — from [`Box::into_raw`], points to
///   [`PeekableReceiver<(Instant, E)>`].
/// * `live_rx_ptr: *mut u8` — from [`Box::into_raw`], points to
///   [`PeekableReceiver<(Instant, E)>`].
/// * `output_ptr: *mut u8` — from [`Box::into_raw`], points to `S::Output`.
pub type InitFn = Box<dyn FnOnce(Instant) -> (*mut u8, *mut u8, *mut u8)>;

/// Type-erased poll function pointer for a source channel.
///
/// # Parameters
///
/// * `rx_ptr: *mut u8` — points to [`PeekableReceiver<(Instant, E)>`].
/// * `cx: &mut Context<'_>` — async task context.
///
/// # Returns
///
/// * `Poll::Ready(Some(ts))` if an event is buffered with timestamp `ts`.
/// * `Poll::Ready(None)` if the channel is closed.
/// * `Poll::Pending` if no event ready; waker registered.
pub type PollFn = unsafe fn(*mut u8, &mut Context<'_>) -> Poll<Option<Instant>>;

/// Type-erased write function pointer for a source.
///
/// # Parameters
///
/// * `rx_ptr: *mut u8` — points to [`PeekableReceiver<(Instant, E)>`].
/// * `output_ptr: *mut u8` — points to `S::Output`.
/// * `timestamp: Instant` — coalesced batch timestamp (overrides event timestamp).
///
/// # Returns
///
/// * `true` if downstream propagation should occur.
pub type WriteFn = unsafe fn(*mut u8, *mut u8, Instant) -> bool;

/// Type-erased representation of a source.
///
/// # Lifecycle
///
/// 1. Created via [`from_source`](ErasedSource::from_source) (safe, typed)
///    or [`new`](ErasedSource::new) (`unsafe`, raw).
/// 2. Consumed by [`Scenario::add_erased_source`], which calls
///    [`init`](ErasedSource::init), constructs the DAG node, and stores
///    channel state in `SourceState`.
pub struct ErasedSource {
    event_type_id: TypeId,
    output_type_id: TypeId,
    estimated_event_count: Option<usize>,
    init_fn: InitFn,
    poll_fn: PollFn,
    write_fn: WriteFn,
    rx_drop_fn: unsafe fn(*mut u8),
    output_drop_fn: unsafe fn(*mut u8),
}

impl ErasedSource {
    /// Construct from a typed [`Source`].
    pub fn from_source<S: Source>(source: S) -> Self {
        let estimated_event_count = source.estimated_event_count();
        Self {
            event_type_id: TypeId::of::<S::Event>(),
            output_type_id: TypeId::of::<S::Output>(),
            estimated_event_count,
            init_fn: Box::new(move |timestamp: Instant| {
                let (hist, live, output) = source.init(timestamp);
                let hist_rx_ptr = Box::into_raw(Box::new(PeekableReceiver::new(hist))) as *mut u8;
                let live_rx_ptr = Box::into_raw(Box::new(PeekableReceiver::new(live))) as *mut u8;
                let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
                (hist_rx_ptr, live_rx_ptr, output_ptr)
            }),
            poll_fn: erased_poll_fn::<S>,
            write_fn: erased_write_fn::<S>,
            rx_drop_fn: erased_drop_fn::<PeekableReceiver<(Instant, S::Event)>>,
            output_drop_fn: erased_drop_fn::<S::Output>,
        }
    }

    /// Construct from raw, type-erased components.
    ///
    /// # Safety
    ///
    /// * `init_fn` must return valid `(hist_rx_ptr, live_rx_ptr, output_ptr)`
    ///   from [`Box::into_raw`] pointing to objects of types
    ///   [`PeekableReceiver<(Instant, E)>`], [`PeekableReceiver<(Instant, E)>`],
    ///   and `output_type_id` respectively, where `E` has type ID
    ///   `event_type_id`.
    /// * `poll_fn` must correctly poll the [`PeekableReceiver<(Instant, E)>`].
    /// * `write_fn` must correctly interpret `rx_ptr` and `output_ptr` as
    ///   pointers to objects of types [`PeekableReceiver<(Instant, E)>`] and
    ///   `output_type_id` respectively.
    /// * `rx_drop_fn` must correctly drop [`Box::from_raw`] pointing to
    ///   an object of type [`PeekableReceiver<(Instant, E)>`].
    /// * `output_drop_fn` must correctly drop [`Box::from_raw`] pointing to
    ///   an object of type `output_type_id`.
    pub unsafe fn new(
        event_type_id: TypeId,
        output_type_id: TypeId,
        estimated_event_count: Option<usize>,
        init_fn: InitFn,
        poll_fn: PollFn,
        write_fn: WriteFn,
        rx_drop_fn: unsafe fn(*mut u8),
        output_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            event_type_id,
            output_type_id,
            estimated_event_count,
            init_fn,
            poll_fn,
            write_fn,
            rx_drop_fn,
            output_drop_fn,
        }
    }

    /// Estimated total number of events this source will emit, if known.
    pub fn estimated_event_count(&self) -> Option<usize> {
        self.estimated_event_count
    }

    /// The [`TypeId`] of the source's event type.
    pub fn event_type_id(&self) -> TypeId {
        self.event_type_id
    }

    /// The [`TypeId`] of the source's output type.
    pub fn output_type_id(&self) -> TypeId {
        self.output_type_id
    }

    /// The type-erased poll function pointer for peeking channel events.
    pub fn poll_fn(&self) -> PollFn {
        self.poll_fn
    }

    /// The type-erased write function pointer for consuming buffered events.
    pub fn write_fn(&self) -> WriteFn {
        self.write_fn
    }

    /// The type-erased drop function for the channel receivers.
    pub fn rx_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.rx_drop_fn
    }

    /// The type-erased drop function for the source's output.
    pub fn output_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.output_drop_fn
    }

    /// Consume the init closure, producing `(hist_rx_ptr, live_rx_ptr, output_ptr)`.
    pub fn init(self, timestamp: Instant) -> (*mut u8, *mut u8, *mut u8) {
        (self.init_fn)(timestamp)
    }
}

/// Type-erased poll function, monomorphised per event type.
unsafe fn erased_poll_fn<S: Source>(
    rx_ptr: *mut u8,
    ctx: &mut Context<'_>,
) -> Poll<Option<Instant>> {
    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(Instant, S::Event)>) };
    rx.poll_pending(ctx).map(|opt| opt.map(|item| item.0))
}

/// Type-erased write function, monomorphised per source type.
unsafe fn erased_write_fn<S: Source>(
    rx_ptr: *mut u8,
    output_ptr: *mut u8,
    timestamp: Instant,
) -> bool {
    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(Instant, S::Event)>) };
    let output = unsafe { &mut *(output_ptr as *mut S::Output) };
    if let Some(item) = rx.take_pending() {
        S::write(item.1, output, timestamp)
    } else {
        false
    }
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
