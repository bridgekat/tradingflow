use std::any::TypeId;
use std::future::Future;
use std::pin::Pin;

use tokio::sync::mpsc;

/// An asynchronous data source that receives events via channels and
/// writes a typed output.
///
/// # Lifecycle
///
/// 1. [`init`](Self::init) consumes the spec, producing two channel
///    receivers (historical + live) and the initial [`Output`](Self::Output).
/// 2. [`write`](Self::write) is called for each received event with the
///    event payload, output reference, and timestamp.
pub trait Source: 'static {
    /// Channel event type.
    type Event: Send + 'static;
    /// Output type.
    type Output: Send + 'static;

    /// Consume the spec and produce channel receivers and initial output.
    fn init(
        self,
        timestamp: i64,
    ) -> (
        mpsc::Receiver<(i64, Self::Event)>,
        mpsc::Receiver<(i64, Self::Event)>,
        Self::Output,
    );

    /// Write an event into the output.
    ///
    /// Returns `true` if downstream propagation should occur.
    fn write(event: Self::Event, output: &mut Self::Output, timestamp: i64) -> bool;
}

/// Type-erased initialization function closure.
///
/// Arguments: `timestamp`.
/// Returns `(hist_receiver, live_receiver, output_ptr)`, where `output_ptr` is
/// from [`Box::into_raw`].
pub type InitFn = Box<dyn FnOnce(i64) -> (ErasedReceiver, ErasedReceiver, *mut u8)>;

/// Type-erased source write function pointer.
///
/// Arguments: `(receiver_state_ptr, output_ptr)`.
/// Returns `true` if downstream propagation should occur.
pub type WriteFn = unsafe fn(*mut u8, *mut u8) -> bool;

/// Type-erased representation of a source.
///
/// # Lifecycle
///
/// 1. Created via [`from_source`](ErasedSource::from_source) (safe, typed)
///    or [`new`](ErasedSource::new) (`unsafe`, raw).
/// 2. Consumed by [`Scenario::add_erased_source`], which calls
///    [`init`](ErasedSource::init), creates the output node, and registers
///    the receivers.
/// 3. If dropped without being installed, the closure drops its captured
///    state automatically.
pub struct ErasedSource {
    event_type_id: TypeId,
    output_type_id: TypeId,
    init_fn: InitFn,
    write_fn: WriteFn,
    output_drop_fn: unsafe fn(*mut u8),
}

impl ErasedSource {
    /// Construct from a typed [`Source`].
    pub fn from_source<S: Source>(source: S) -> Self {
        Self {
            event_type_id: TypeId::of::<S::Event>(),
            output_type_id: TypeId::of::<S::Output>(),
            init_fn: Box::new(move |timestamp: i64| {
                let (hist_rx, live_rx, output) = source.init(timestamp);
                let hist = ErasedReceiver::from_receiver(hist_rx);
                let live = ErasedReceiver::from_receiver(live_rx);
                let output_ptr = Box::into_raw(Box::new(output)) as *mut u8;
                (hist, live, output_ptr)
            }),
            write_fn: erased_write_fn::<S>,
            output_drop_fn: erased_drop_fn::<S::Output>,
        }
    }

    /// Construct from raw, type-erased components.
    ///
    /// # Safety
    ///
    /// * `init_fn` must return valid receivers and a valid output pointer
    ///   from `Box::into_raw` for the `output_type_id` type.
    /// * `write_fn` must correctly extract the pending event from the
    ///   receiver state and write it into the output.
    /// * `output_drop_fn` must correctly drop the output type.
    pub unsafe fn new(
        event_type_id: TypeId,
        output_type_id: TypeId,
        init_fn: InitFn,
        write_fn: WriteFn,
        output_drop_fn: unsafe fn(*mut u8),
    ) -> Self {
        Self {
            event_type_id,
            output_type_id,
            init_fn,
            write_fn,
            output_drop_fn,
        }
    }

    pub fn event_type_id(&self) -> TypeId {
        self.event_type_id
    }

    pub fn output_type_id(&self) -> TypeId {
        self.output_type_id
    }

    pub fn write_fn(&self) -> WriteFn {
        self.write_fn
    }

    pub fn output_drop_fn(&self) -> unsafe fn(*mut u8) {
        self.output_drop_fn
    }

    /// Consume the source, producing `(hist_receiver, live_receiver, output_ptr)`.
    pub fn init(self, timestamp: i64) -> (ErasedReceiver, ErasedReceiver, *mut u8) {
        (self.init_fn)(timestamp)
    }
}

/// Type-erased write function, monomorphised per source type.
unsafe fn erased_write_fn<S: Source>(state_ptr: *mut u8, output_ptr: *mut u8) -> bool {
    let state = unsafe { &mut *(state_ptr as *mut ReceiverState<S::Event>) };
    let output = unsafe { &mut *(output_ptr as *mut S::Output) };
    if let Some((ts, event)) = state.pending.take() {
        S::write(event, output, ts)
    } else {
        false
    }
}

/// Type-erased async wait function pointer.
///
/// Arguments: `receiver_state_ptr`.
/// Returns a future that resolves to `Some(timestamp)` when an event is ready, or
/// `None` when the channel is closed.
type WaitFn = unsafe fn(*mut u8) -> Pin<Box<dyn Future<Output = Option<i64>>>>;

/// Type-erased try wait function pointer.
///
/// Arguments: `receiver_state_ptr`.
/// Returns `Some(Some(timestamp))` when an event is ready,
/// `Some(None)` when not ready, or `None` when the channel is closed.
type TryWaitFn = unsafe fn(*mut u8) -> Option<Option<i64>>;

/// Type-erased async event receiver.
///
/// Wraps a [`ReceiverState`] behind a raw pointer with monomorphised
/// function pointers for async polling.
pub struct ErasedReceiver {
    state: *mut u8,
    wait_fn: WaitFn,
    try_wait_fn: TryWaitFn,
    state_drop_fn: unsafe fn(*mut u8),
}

// SAFETY: `state` owns the allocation `ReceiverState<E>: Send`.
unsafe impl Send for ErasedReceiver {}

impl ErasedReceiver {
    /// Construct from a typed [`mpsc::Receiver`].
    pub fn from_receiver<E: Send + 'static>(rx: mpsc::Receiver<(i64, E)>) -> Self {
        let state = Box::into_raw(Box::new(ReceiverState::<E> { rx, pending: None }));
        Self {
            state: state as *mut u8,
            wait_fn: erased_wait::<E>,
            try_wait_fn: erased_try_wait::<E>,
            state_drop_fn: erased_drop_fn::<ReceiverState<E>>,
        }
    }

    /// Block until the next event is available and return its timestamp.
    /// Returns `None` when the channel is closed.
    pub fn wait(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>> {
        // SAFETY: self owns the state; &mut self keeps it alive for the
        // future's lifetime.  The 'static future from wait_fn is safely
        // narrowed to '_.
        unsafe { (self.wait_fn)(self.state) }
    }

    /// Non-blocking poll.
    /// `Some(Some(ts))` — event pending; `Some(None)` — not ready; `None` — closed.
    pub fn try_wait(&mut self) -> Option<Option<i64>> {
        unsafe { (self.try_wait_fn)(self.state) }
    }

    /// Raw pointer to the receiver state.
    pub fn state_ptr(&mut self) -> *mut u8 {
        self.state
    }
}

impl Drop for ErasedReceiver {
    fn drop(&mut self) {
        unsafe { (self.state_drop_fn)(self.state) };
    }
}

/// Receiver state: mpsc receiver + one-slot pending buffer.
pub struct ReceiverState<E: Send + 'static> {
    pub rx: mpsc::Receiver<(i64, E)>,
    pub pending: Option<(i64, E)>,
}

unsafe fn erased_wait<E: Send + 'static>(
    state: *mut u8,
) -> Pin<Box<dyn Future<Output = Option<i64>>>> {
    Box::pin(async move {
        let rs = unsafe { &mut *(state as *mut ReceiverState<E>) };
        if let Some(ref item) = rs.pending {
            return Some(item.0);
        }
        rs.rx.recv().await.map(|item| {
            let ts = item.0;
            rs.pending = Some(item);
            ts
        })
    })
}

unsafe fn erased_try_wait<E: Send + 'static>(state: *mut u8) -> Option<Option<i64>> {
    let rs = unsafe { &mut *(state as *mut ReceiverState<E>) };
    if let Some(ref item) = rs.pending {
        return Some(Some(item.0));
    }
    match rs.rx.try_recv() {
        Ok(item) => {
            let ts = item.0;
            rs.pending = Some(item);
            Some(Some(ts))
        }
        Err(mpsc::error::TryRecvError::Empty) => Some(None),
        Err(mpsc::error::TryRecvError::Disconnected) => None,
    }
}

/// Type-erased box drop function, monomorphised per value type.
unsafe fn erased_drop_fn<T>(ptr: *mut u8) {
    unsafe { drop(Box::from_raw(ptr as *mut T)) };
}
