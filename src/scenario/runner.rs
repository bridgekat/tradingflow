//! Async source handling and the POCQ event loop.
//!
//! [`SourceState`] tracks per-source runtime state.  The [`Scenario::run`]
//! method implements the Point-of-Coherency Queue (POCQ) algorithm that
//! consumes all registered sources and propagates events through the DAG.

use crate::source::{ErasedReceiver, WriteFn};

use super::Scenario;

// ---------------------------------------------------------------------------
// SourceState
// ---------------------------------------------------------------------------

/// Per-source runtime state for [`Scenario`].
///
/// Each source has independent historical and live receivers.  Both can
/// have a pending event simultaneously.
pub(super) struct SourceState {
    pub node_index: usize,
    hist: ErasedReceiver,
    live: ErasedReceiver,
    write_fn: WriteFn,
    hist_ts: Option<i64>,
    live_ts: Option<i64>,
    hist_exhausted: bool,
    live_exhausted: bool,
}

impl SourceState {
    pub(super) fn new(
        node_index: usize,
        hist: ErasedReceiver,
        live: ErasedReceiver,
        write_fn: WriteFn,
    ) -> Self {
        Self {
            node_index,
            hist,
            live,
            write_fn,
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
// Scenario — POCQ event loop
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
                        Some(ts) => src.live_ts = ts,
                        None => src.live_exhausted = true,
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
                let value_ptr = self.graph.nodes[src.node_index].value;
                if src.hist_ts == Some(min_ts) {
                    unsafe { (src.write_fn)(src.hist.state_ptr(), value_ptr) };
                    queue_sources.push(src.node_index);
                    src.hist_ts = None;
                }
                if src.live_ts == Some(min_ts) {
                    unsafe { (src.write_fn)(src.live.state_ptr(), value_ptr) };
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
    async fn wait_any_live(&mut self) {
        loop {
            for src in &mut self.source_states {
                if src.live_exhausted || src.live_ts.is_some() {
                    continue;
                }
                match src.live.try_wait() {
                    Some(Some(ts)) => {
                        src.live_ts = Some(ts);
                        return;
                    }
                    Some(None) => {}
                    None => src.live_exhausted = true,
                }
            }
            if self
                .source_states
                .iter()
                .all(|s| s.live_exhausted || s.live_ts.is_some())
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    }
}
