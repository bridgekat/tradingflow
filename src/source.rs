use std::any::TypeId;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

use crate::Instant;
use crate::PeekableReceiver;

/// An asynchronous data source that receives events via channels and writes
/// a typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) reads the spec by reference and produces two
///    channel receivers (historical + live) plus the initial
///    [`Output`](Self::Output).
/// 2. [`write`](Self::write) is called for each received event to update the
///    output.
///
/// # Reusability
///
/// `init` takes `&self`, so a single spec can drive multiple scenario
/// sessions - the type-erasure layer ([`ErasedSource`]) keeps the spec by
/// value and calls `init` against the shared reference on every session
/// start.  Implementations should treat the spec as immutable
/// configuration; any per-session state lives in
/// [`Output`](Self::Output) (built fresh by `init`) and any per-event
/// state is threaded through `Output` as well.  Clone any field that
/// needs to move into an async driver task (e.g. into
/// [`tokio::spawn`]) explicitly.
pub trait Source: 'static {
    /// Channel event type.
    type Event: Send + 'static;
    /// Output type.
    type Output: Send + 'static;
    /// Per-source mutable state threaded through every [`write`](Self::write)
    /// call (created by [`init`](Self::init)). Use `()` for stateless sources;
    /// sources that need to remember something across events — e.g. a panel that
    /// clears the previous tick's entries when the timestamp advances — keep it
    /// here. Lives on the driver thread, so no `Sync` bound.
    type State: Send + 'static;

    /// Build channel receivers, the initial output, and the initial write
    /// [`State`](Self::State) from a borrow of the spec.
    #[allow(clippy::type_complexity)]
    fn init(
        &self,
        timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, Self::Event)>,
        mpsc::Receiver<(Instant, Self::Event)>,
        Self::Output,
        Self::State,
    );

    /// Apply a received channel item to the output, threading `state`, and return
    /// **how many logical events it represents** (for the run's event count).
    ///
    /// A source may batch many events into one channel item (e.g. a panel ships a
    /// whole tick's rows as one `Vec`); `write` then applies them **per event**
    /// (the per-event state logic still runs for each) and returns the batch size.
    /// A one-event-per-item source returns `1`. The engine marks the output cell's
    /// cone regardless of the return.
    fn write(
        state: &mut Self::State,
        event: Self::Event,
        output: &mut Self::Output,
        timestamp: Instant,
    ) -> usize;

    /// Estimated total number of events this source will emit over its
    /// lifetime.  `None` for live / unbounded sources.
    ///
    /// Used only for progress reporting via
    /// [`Scenario::run`](crate::Scenario::run)'s `on_flush` callback -
    /// treated as advisory.  The default returns `None`.
    fn estimated_event_count(&self) -> Option<usize> {
        None
    }
}

/// Type-erased initialization closure for a source.
///
/// `Fn` rather than `FnOnce`: an [`ErasedSource`] is a *definition* that
/// can be re-initialised any number of times.  Each call allocates a fresh
/// pair of channel receivers and a fresh output, and (for typed sources
/// constructed via [`ErasedSource::from_source`]) clones the captured spec
/// before consuming it via [`Source::init`].
///
/// # Parameters
///
/// * `timestamp: Instant` - initial timestamp.
///
/// # Returns
///
/// * `hist_rx_ptr: *mut u8` - from [`Box::into_raw`], points to
///   [`PeekableReceiver<(Instant, E)>`].
/// * `live_rx_ptr: *mut u8` - from [`Box::into_raw`], points to
///   [`PeekableReceiver<(Instant, E)>`].
/// * `output_ptr: *mut u8` - from [`Box::into_raw`], points to `S::Output`.
/// * `state_ptr: *mut u8` - from [`Box::into_raw`], points to `S::State`.
pub type InitFn = Box<dyn Fn(Instant) -> (*mut u8, *mut u8, *mut u8, *mut u8)>;

/// Type-erased poll function pointer for a source channel.
///
/// # Parameters
///
/// * `rx_ptr: *mut u8` - points to [`PeekableReceiver<(Instant, E)>`].
/// * `cx: &mut Context<'_>` - async task context.
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
/// * `state_ptr: *mut u8` - points to `S::State`.
/// * `rx_ptr: *mut u8` - points to [`PeekableReceiver<(Instant, E)>`].
/// * `output_ptr: *mut u8` - points to `S::Output`.
/// * `timestamp: Instant` - coalesced batch timestamp (overrides event timestamp).
///
/// # Returns
///
/// * the number of logical events the consumed channel item represented
///   (a batched source returns its batch size); `0` if nothing was buffered.
pub type WriteFn = unsafe fn(*mut u8, *mut u8, *mut u8, Instant) -> usize;

/// Type-erased representation of a source.
///
/// # Lifecycle
///
/// 1. Created via [`from_source`](ErasedSource::from_source) (safe, typed)
///    or [`new`](ErasedSource::new) (`unsafe`, raw).
/// 2. Held by `Scenario::add_erased_source`, which borrows it to call
///    [`init`](ErasedSource::init) and construct the graph node, storing
///    channel state in `SourceState`.  Because [`init`](ErasedSource::init)
///    takes `&self`, the same `ErasedSource` can drive multiple initialisations
///    - one per scenario session - without being consumed.
pub struct ErasedSource {
    event_type_id: TypeId,
    output_type_id: TypeId,
    estimated_event_count: Option<usize>,
    init_fn: InitFn,
    poll_fn: PollFn,
    write_fn: WriteFn,
    rx_drop_fn: unsafe fn(*mut u8),
    output_drop_fn: unsafe fn(*mut u8),
    state_drop_fn: unsafe fn(*mut u8),
}

impl ErasedSource {
    /// Construct from a typed [`Source`].
    ///
    /// The spec is captured by value into the [`InitFn`] closure and
    /// borrowed by every call to [`Source::init`] - no `Clone` bound is
    /// required on `S`, since `init` takes `&self`.  Implementations that
    /// need to move owned data into spawned tasks should clone the
    /// relevant fields out of `&self` themselves.
    pub fn from_source<S: Source>(source: S) -> Self {
        let estimated_event_count = source.estimated_event_count();
        Self {
            event_type_id: TypeId::of::<S::Event>(),
            output_type_id: TypeId::of::<S::Output>(),
            estimated_event_count,
            init_fn: Box::new(move |timestamp: Instant| {
                let (hist, live, output, state) = source.init(timestamp);
                let hist_rx_ptr = Box::into_raw(Box::new(PeekableReceiver::new(hist))) as *mut u8;
                let live_rx_ptr = Box::into_raw(Box::new(PeekableReceiver::new(live))) as *mut u8;
                let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
                let state_ptr = Box::into_raw(Box::new(state)) as *mut u8;
                (hist_rx_ptr, live_rx_ptr, output_ptr, state_ptr)
            }),
            poll_fn: erased_poll_fn::<S>,
            write_fn: erased_write_fn::<S>,
            rx_drop_fn: erased_drop_fn::<PeekableReceiver<(Instant, S::Event)>>,
            output_drop_fn: erased_drop_fn::<S::Output>,
            state_drop_fn: erased_drop_fn::<S::State>,
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
        state_drop_fn: unsafe fn(*mut u8),
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
            state_drop_fn,
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

    /// The type-erased drop function for the source's write state.
    pub fn state_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.state_drop_fn
    }

    /// Invoke the init closure, producing `(hist_rx_ptr, live_rx_ptr, output_ptr, state_ptr)`.
    ///
    /// Takes `&self` rather than `self` - each call allocates a fresh set of
    /// channel receivers, output, and write state, so an [`ErasedSource`] can
    /// drive multiple sessions over its lifetime.
    pub fn init(&self, timestamp: Instant) -> (*mut u8, *mut u8, *mut u8, *mut u8) {
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
    state_ptr: *mut u8,
    rx_ptr: *mut u8,
    output_ptr: *mut u8,
    timestamp: Instant,
) -> usize {
    let state = unsafe { &mut *(state_ptr as *mut S::State) };
    let rx = unsafe { &mut *(rx_ptr as *mut PeekableReceiver<(Instant, S::Event)>) };
    let output = unsafe { &mut *(output_ptr as *mut S::Output) };
    if let Some(item) = rx.take_pending() {
        S::write(state, item.1, output, timestamp)
    } else {
        0
    }
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial stateless source whose [`init`](Source::init) reads the
    /// captured spec by reference.  Used to exercise reusability of
    /// [`ErasedSource`] without requiring `Clone`.
    struct FixedSource {
        value: i64,
    }

    impl Source for FixedSource {
        type Event = i64;
        type Output = i64;
        type State = ();

        fn init(
            &self,
            _timestamp: Instant,
        ) -> (
            tokio::sync::mpsc::Receiver<(Instant, i64)>,
            tokio::sync::mpsc::Receiver<(Instant, i64)>,
            i64,
            (),
        ) {
            let (_, hist_rx) = tokio::sync::mpsc::channel(1);
            let (_, live_rx) = tokio::sync::mpsc::channel(1);
            (hist_rx, live_rx, self.value, ())
        }

        fn write(_state: &mut (), event: i64, output: &mut i64, _timestamp: Instant) -> usize {
            *output = event;
            1
        }
    }

    /// `ErasedSource::init` takes `&self`, so the descriptor can drive
    /// multiple sessions; each call yields fresh receivers and a fresh
    /// output cell.
    #[test]
    fn erased_source_can_reinit() {
        let erased = ErasedSource::from_source(FixedSource { value: 42 });
        let output_drop = erased.output_drop_fn();
        let rx_drop = erased.rx_drop_fn();
        let state_drop = erased.state_drop_fn();

        for _ in 0..3 {
            let (hist_ptr, live_ptr, output_ptr, state_ptr) = erased.init(Instant::MIN);
            assert!(!hist_ptr.is_null());
            assert!(!live_ptr.is_null());
            assert_eq!(unsafe { *(output_ptr as *const i64) }, 42);
            unsafe {
                rx_drop(hist_ptr);
                rx_drop(live_ptr);
                output_drop(output_ptr);
                state_drop(state_ptr);
            }
        }
    }
}
