//! Async source handling and the POCQ event loop.
//!
//! [`ErasedChannel`] provides type-erased channel operations for sources.
//! [`SourceState`] tracks per-source runtime state.  The [`Scenario::run`]
//! method implements the Point-of-Coherency Queue (POCQ) algorithm that
//! consumes all registered sources and propagates events through the DAG.

use std::future::Future;
use std::pin::Pin;

use crate::source::Source;
use crate::store::Store;

use super::Scenario;

// ---------------------------------------------------------------------------
// ErasedChannel — one trait for both historical and live channels
// ---------------------------------------------------------------------------

/// Type-erased async channel for source events.
///
/// Both historical and live channels implement this same trait.  The
/// scheduling policy (historical constraint vs free-running) is in the
/// POCQ loop, not here.
trait ErasedChannel: Send {
    /// Block until the next event is available and return its timestamp.
    ///
    /// If an event is already pending, returns its timestamp immediately.
    /// Returns `None` when the channel is closed (source exhausted).
    fn wait(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>>;

    /// Non-blocking poll for the next event.
    ///
    /// * `Ok(Some(ts))` — event pending with this timestamp.
    /// * `Ok(None)` — no event ready right now, channel still open.
    /// * `Err(())` — channel closed (source exhausted).
    fn try_wait(&mut self) -> Result<Option<i64>, ()>;

    /// Write the pending event to a store via [`Store::push_default`] +
    /// [`Store::element_view_mut`].
    ///
    /// The event's own timestamp is used for the new element.
    /// Returns `true` if a value was produced.
    ///
    /// # Safety
    ///
    /// `store_ptr` must point to a valid `Store<S::Scalar>`.
    unsafe fn write(&mut self, store_ptr: *mut u8) -> bool;
}

/// Concrete [`ErasedChannel`] for a given [`Source`].
struct TypedChannel<S: Source> {
    rx: tokio::sync::mpsc::Receiver<(i64, S::Event)>,
    pending: Option<(i64, S::Event)>,
}

impl<S: Source> TypedChannel<S> {
    fn fill_pending(&mut self, item: (i64, S::Event)) -> i64 {
        let ts = item.0;
        self.pending = Some(item);
        ts
    }
}

impl<S: Source> ErasedChannel for TypedChannel<S> {
    fn wait(&mut self) -> Pin<Box<dyn Future<Output = Option<i64>> + '_>> {
        Box::pin(async move {
            if let Some(ref item) = self.pending {
                return Some(item.0);
            }
            self.rx.recv().await.map(|item| self.fill_pending(item))
        })
    }

    fn try_wait(&mut self) -> Result<Option<i64>, ()> {
        if let Some(ref item) = self.pending {
            return Ok(Some(item.0));
        }
        match self.rx.try_recv() {
            Ok(item) => Ok(Some(self.fill_pending(item))),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => Err(()),
        }
    }

    unsafe fn write(&mut self, store_ptr: *mut u8) -> bool {
        if let Some((ts, payload)) = self.pending.take() {
            let store = unsafe { &mut *(store_ptr as *mut Store<S::Scalar>) };
            store.push_default(ts);
            let view = store.current_view_mut();
            let produced = S::write(payload, view);
            if produced {
                store.commit();
            } else {
                store.rollback();
            }
            produced
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// SourceState
// ---------------------------------------------------------------------------

/// Per-source runtime state for [`Scenario`].
///
/// Each source has independent historical and live channels.  Both can
/// have a pending event simultaneously.
pub(super) struct SourceState {
    pub node_index: usize,
    hist: Box<dyn ErasedChannel>,
    live: Box<dyn ErasedChannel>,
    hist_ts: Option<i64>,
    live_ts: Option<i64>,
    hist_exhausted: bool,
    live_exhausted: bool,
}

impl SourceState {
    /// Create a new `SourceState` from typed channel receivers.
    pub(super) fn new<S: Source>(
        node_index: usize,
        hist_rx: tokio::sync::mpsc::Receiver<(i64, S::Event)>,
        live_rx: tokio::sync::mpsc::Receiver<(i64, S::Event)>,
    ) -> Self {
        Self {
            node_index,
            hist: Box::new(TypedChannel::<S> {
                rx: hist_rx,
                pending: None,
            }),
            live: Box::new(TypedChannel::<S> {
                rx: live_rx,
                pending: None,
            }),
            hist_ts: None,
            live_ts: None,
            hist_exhausted: false,
            live_exhausted: false,
        }
    }

    /// The minimum pending timestamp across both channels.
    fn min_pending_ts(&self) -> Option<i64> {
        match (self.hist_ts, self.live_ts) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario — source registration and unified POCQ
// ---------------------------------------------------------------------------

impl Scenario {
    /// Run the unified POCQ event loop.
    ///
    /// Historical and live events participate in the same loop.  The
    /// **historical constraint** (all non-exhausted historical channels must
    /// have a pending event) is enforced before each scheduling decision.
    /// Live events are polled non-blocking alongside.  The global minimum
    /// timestamp across ALL pending events (historical and live) determines
    /// what gets processed next.
    ///
    /// This means a live event with a smaller timestamp than all pending
    /// historical events is processed first — correctly handling sources
    /// that start at different times.
    pub async fn run(&mut self) {
        let mut queue_ts: Option<i64> = None;
        let mut queue_sources: Vec<usize> = Vec::new();

        loop {
            // ----------------------------------------------------------
            // Step 1: Historical constraint — block until every active
            // historical channel has a pending event.
            // ----------------------------------------------------------
            for src in &mut self.source_states {
                if !src.hist_exhausted && src.hist_ts.is_none() {
                    let ts = src.hist.wait().await;
                    src.hist_ts = ts;
                    if ts.is_none() {
                        src.hist_exhausted = true;
                    }
                }
            }

            // ----------------------------------------------------------
            // Step 2: Poll live channels (non-blocking).
            // ----------------------------------------------------------
            for src in &mut self.source_states {
                if !src.live_exhausted && src.live_ts.is_none() {
                    match src.live.try_wait() {
                        Ok(ts) => src.live_ts = ts,
                        Err(()) => src.live_exhausted = true,
                    }
                }
            }

            // ----------------------------------------------------------
            // Step 3: Find global minimum across all pending events.
            // ----------------------------------------------------------
            let min_ts = self
                .source_states
                .iter()
                .filter_map(|s| s.min_pending_ts())
                .min();

            let Some(min_ts) = min_ts else {
                // No pending events.  If all live channels are also
                // exhausted, we're done.
                let any_live = self.source_states.iter().any(|s| !s.live_exhausted);
                if !any_live {
                    break;
                }
                // Block until any live channel produces an event.
                self.wait_any_live().await;
                continue;
            };

            // ----------------------------------------------------------
            // Step 4: Coalesce — flush if timestamp advances.
            // ----------------------------------------------------------
            if let Some(qts) = queue_ts
                && min_ts > qts
            {
                self.graph.flush(qts, &queue_sources);
                queue_sources.clear();
            }

            // ----------------------------------------------------------
            // Step 5: Collect all events at min_ts from both channels.
            // ----------------------------------------------------------
            for src in &mut self.source_states {
                let store_ptr = self.graph.nodes[src.node_index].store;
                if src.hist_ts == Some(min_ts) {
                    // SAFETY: store_ptr is valid (node invariant).
                    unsafe { src.hist.write(store_ptr) };
                    queue_sources.push(src.node_index);
                    src.hist_ts = None;
                }
                if src.live_ts == Some(min_ts) {
                    unsafe { src.live.write(store_ptr) };
                    queue_sources.push(src.node_index);
                    src.live_ts = None;
                }
            }
            queue_ts = Some(min_ts);
        }

        // Final flush.
        if !queue_sources.is_empty() {
            self.graph.flush(queue_ts.unwrap(), &queue_sources);
        }
    }

    /// Block until any non-exhausted live channel produces an event.
    ///
    /// Uses sequential polling with yield.  A future optimisation could
    /// use `futures::future::select_all` for true concurrency.
    async fn wait_any_live(&mut self) {
        loop {
            for src in &mut self.source_states {
                if src.live_exhausted || src.live_ts.is_some() {
                    continue;
                }
                match src.live.try_wait() {
                    Ok(Some(ts)) => {
                        src.live_ts = Some(ts);
                        return;
                    }
                    Ok(None) => {} // no event yet, channel still open
                    Err(()) => src.live_exhausted = true,
                }
            }
            // Check if all live channels are done.
            if self
                .source_states
                .iter()
                .all(|s| s.live_exhausted || s.live_ts.is_some())
            {
                return;
            }
            // Yield to the runtime and retry.
            tokio::task::yield_now().await;
        }
    }
}
