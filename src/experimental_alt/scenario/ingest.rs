//! Source-ingest event loop — adapted from `src/scenario/queue.rs`.
//!
//! The hist/live merge, `drain_hist` early-exit, live-timestamp clamping,
//! and coalescing semantics are identical to the existing runtime.  The
//! only behavioural difference is that instead of synchronously calling
//! `graph.flush(ts, &firing_sources)` at each batch boundary, this
//! adapted version:
//!
//! 1. Obtains the next `tick_no` and blocks while more than
//!    [`pipeline_width`](crate::experimental_alt::scenario::Scenario::pipeline_width)
//!    ticks are in flight (so the per-node `tick mod W` slot is free).
//! 2. Resets every wavefront-participant operator's `remaining_inputs`
//!    and `incoming_bits` for this tick slot to the node's
//!    `effective_upstream_count`.
//! 3. For each firing source: writes the event into a fresh output
//!    buffer, commits it to the source's queue, and fires trigger edges
//!    into downstreams (setting their input bits and decrementing their
//!    remaining-inputs counters; if a counter hits 0 the downstream task
//!    is enqueued onto the scheduler).
//! 4. For each non-firing source: fires trigger edges into downstreams
//!    **without** setting the bit — the counter still decrements so the
//!    downstream eventually runs (and sees no-fire on that input).

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::future::poll_fn;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::Poll;
use std::time::Duration;

use futures::stream::{FuturesUnordered, StreamExt};

use super::super::data::Instant;
use super::super::operator::NodeKind;
use super::super::source::PollFn;
use super::graph::Graph;
use super::scheduler::{SharedState, Task};

// ---------------------------------------------------------------------------
// Channel-kind and per-source receive future
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Eq, PartialEq)]
enum ChannelKind {
    Hist,
    Live,
}

struct ErasedRecvFuture {
    rx_ptr: *mut u8,
    poll_fn: PollFn,
    source_idx: usize,
    kind: ChannelKind,
}

unsafe impl Send for ErasedRecvFuture {}

impl std::future::Future for ErasedRecvFuture {
    type Output = (usize, ChannelKind, Option<Instant>);

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Self::Output> {
        let src = self.source_idx;
        let kind = self.kind;
        unsafe { (self.poll_fn)(self.rx_ptr, cx) }.map(|opt| (src, kind, opt))
    }
}

fn make_recv_future(graph: &Graph, src_idx: usize, kind: ChannelKind) -> ErasedRecvFuture {
    let n = &graph.nodes[src_idx];
    let rx_ptr = match kind {
        ChannelKind::Hist => n.source_hist_rx_ptr.0,
        ChannelKind::Live => n.source_live_rx_ptr.0,
    };
    let poll_fn = n.source_poll_fn.expect("source node must have poll_fn");
    ErasedRecvFuture {
        rx_ptr,
        poll_fn,
        source_idx: src_idx,
        kind,
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct HeapEntry {
    ts: Instant,
    source_idx: usize,
    kind: ChannelKind,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ts
            .cmp(&other.ts)
            .then_with(|| self.source_idx.cmp(&other.source_idx))
            .then_with(|| (self.kind as u8).cmp(&(other.kind as u8)))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// drain_hist / drain_live — verbatim from src/scenario/queue.rs
// ---------------------------------------------------------------------------

async fn drain_hist(
    hist_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    last_hist_ts: &mut [Instant],
    hist_pending_ts: &mut BTreeMap<Instant, usize>,
) {
    while let Some((src, _kind, opt_ts)) = hist_refills.next().await {
        let lower = last_hist_ts[src];
        if let Some(cnt) = hist_pending_ts.get_mut(&lower) {
            *cnt -= 1;
            if *cnt == 0 {
                hist_pending_ts.remove(&lower);
            }
        }

        if let Some(ts) = opt_ts {
            last_hist_ts[src] = ts;
            heap.push(Reverse(HeapEntry {
                ts,
                source_idx: src,
                kind: ChannelKind::Hist,
            }));
        }

        if hist_refills.is_empty() {
            break;
        }

        let heap_min = heap.peek().map(|&Reverse(e)| e.ts).unwrap_or(Instant::MAX);
        let pending_lower = hist_pending_ts
            .keys()
            .next()
            .copied()
            .unwrap_or(Instant::MAX);
        if pending_lower > heap_min {
            break;
        }
    }
}

async fn drain_live(
    live_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    current_ts: Instant,
) {
    poll_fn(|cx| {
        loop {
            match Pin::new(&mut *live_refills).poll_next_unpin(cx) {
                Poll::Ready(Some((src, kind, Some(ts)))) => {
                    let ts = ts.max(current_ts);
                    heap.push(Reverse(HeapEntry {
                        ts,
                        source_idx: src,
                        kind,
                    }));
                }
                Poll::Ready(Some((_src, _kind, None))) => {
                    // Channel closed.
                }
                Poll::Ready(None) | Poll::Pending => return Poll::Ready(()),
            }
        }
    })
    .await;
}

// ---------------------------------------------------------------------------
// Ingest main loop
// ---------------------------------------------------------------------------

pub(crate) async fn ingest_main(state: &SharedState) {
    let graph: &Graph = &state.graph;
    let n_sources = graph.source_indices.len();
    if n_sources == 0 {
        return;
    }

    let pipeline_width = state.pipeline_width;

    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    let mut hist_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
    let mut live_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
    let mut current_ts: Instant = Instant::MIN;

    let mut last_hist_ts: Vec<Instant> = vec![Instant::MIN; n_sources];
    let mut hist_pending_ts: BTreeMap<Instant, usize> = BTreeMap::new();

    for (i, &_src_node_idx) in graph.source_indices.iter().enumerate() {
        hist_refills.push(make_recv_future(graph, graph.source_indices[i], ChannelKind::Hist));
        *hist_pending_ts.entry(Instant::MIN).or_insert(0) += 1;
        live_refills.push(make_recv_future(graph, graph.source_indices[i], ChannelKind::Live));
    }

    drain_hist(
        &mut hist_refills,
        &mut heap,
        &mut last_hist_ts,
        &mut hist_pending_ts,
    )
    .await;
    drain_live(&mut live_refills, &mut heap, current_ts).await;

    // Coalesce events per-timestamp.
    let mut queue_ts: Option<Instant> = None;
    // firing[src_idx_in_source_indices] = bool (firing this tick)
    let mut firing: Vec<bool> = vec![false; n_sources];
    // Per-source event "taken" bit for this tick.
    let mut next_tick_no: u64 = 0;

    loop {
        drain_hist(
            &mut hist_refills,
            &mut heap,
            &mut last_hist_ts,
            &mut hist_pending_ts,
        )
        .await;
        drain_live(&mut live_refills, &mut heap, current_ts).await;

        let Some(&Reverse(HeapEntry { ts: min_ts, .. })) = heap.peek() else {
            if live_refills.is_empty() {
                break;
            }
            while let Some((src, kind, opt_ts)) = live_refills.next().await {
                if let Some(ts) = opt_ts {
                    let ts = ts.max(current_ts);
                    heap.push(Reverse(HeapEntry {
                        ts,
                        source_idx: src,
                        kind,
                    }));
                    break;
                }
                if live_refills.is_empty() {
                    break;
                }
            }
            continue;
        };

        // On tick boundary, flush the previous tick (if any).
        if let Some(qts) = queue_ts
            && min_ts > qts
        {
            let tick_no = next_tick_no;
            next_tick_no += 1;
            dispatch_tick(state, graph, qts, tick_no, pipeline_width, &firing).await;
            for f in firing.iter_mut() {
                *f = false;
            }
            if state.shutdown.load(Ordering::Acquire) {
                return;
            }
        }

        // Pop every heap entry at min_ts into the coalesced batch.
        while let Some(&Reverse(HeapEntry { ts, .. })) = heap.peek() {
            if ts > min_ts {
                break;
            }
            let Reverse(HeapEntry {
                ts: _,
                source_idx,
                kind,
            }) = heap.pop().unwrap();

            // Consume the peeked event by calling write_fn, which also
            // commits the slot via the source's output store.
            firing[source_idx] = true;
            commit_source_event(graph, source_idx, kind, min_ts);

            // Re-queue the consumed channel's future.
            match kind {
                ChannelKind::Hist => {
                    last_hist_ts[source_idx] = min_ts;
                    *hist_pending_ts.entry(min_ts).or_insert(0) += 1;
                    hist_refills.push(make_recv_future(
                        graph,
                        graph.source_indices[source_idx],
                        ChannelKind::Hist,
                    ));
                }
                ChannelKind::Live => {
                    live_refills.push(make_recv_future(
                        graph,
                        graph.source_indices[source_idx],
                        ChannelKind::Live,
                    ));
                }
            }
        }
        current_ts = min_ts;
        queue_ts = Some(min_ts);
    }

    // Flush the final accumulated tick.
    if let Some(qts) = queue_ts {
        let tick_no = next_tick_no;
        dispatch_tick(state, graph, qts, tick_no, pipeline_width, &firing).await;
    }
}

// ---------------------------------------------------------------------------
// Source event commit — consume one buffered event into the source's queue
// ---------------------------------------------------------------------------

/// Write the peeked event for the given source+channel into a fresh
/// output buffer and commit it to the source's output queue.  Called
/// after [`make_recv_future`] has resolved with `Some(ts)` and before the
/// per-tick dispatch fires downstream.
fn commit_source_event(graph: &Graph, source_idx: usize, kind: ChannelKind, ts: Instant) {
    let node = &graph.nodes[graph.source_indices[source_idx]];
    debug_assert_eq!(node.kind, NodeKind::Source);
    let rx_ptr = match kind {
        ChannelKind::Hist => node.source_hist_rx_ptr.0,
        ChannelKind::Live => node.source_live_rx_ptr.0,
    };
    let write_fn = node
        .source_write_fn
        .expect("source node must have write_fn");
    let fresh_ptr = node.output.alloc_fresh();
    let ok = unsafe { write_fn(rx_ptr, fresh_ptr, ts) };
    if ok {
        unsafe {
            node.output.commit(ts, fresh_ptr);
        }
        node.last_committed_ts.store(ts.as_nanos(), Ordering::Release);
    } else {
        unsafe {
            node.output.drop_fresh(fresh_ptr);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch one tick's worth of work to the scheduler
// ---------------------------------------------------------------------------

async fn dispatch_tick(
    state: &SharedState,
    graph: &Graph,
    ts: Instant,
    tick_no: u64,
    pipeline_width: usize,
    firing: &[bool],
) {
    // Throttle: make sure the (tick_no mod W) slot is available.  We
    // require every wavefront participant to have already committed tick
    // `tick_no - W` — then its tick-slot is free for reuse.
    if tick_no >= pipeline_width as u64 {
        let required = (tick_no - pipeline_width as u64) as i64;
        wait_for_commit(graph, required).await;
    }

    // Reset per-tick bookkeeping for every wavefront-participant
    // operator.  Source bookkeeping is None — nothing to reset.
    let tick_slot = (tick_no as usize) % pipeline_width;
    for node in graph.nodes.iter() {
        if let Some(bk) = &node.bookkeeping {
            bk.reset(tick_slot, node.effective_upstream_count);
        }
    }

    // Fire source trigger-edges.  Sources with `firing[i] == true`
    // commit-and-fire; others signal "no-fire" (decrement counters
    // only).
    for (i, &_src_node_idx) in graph.source_indices.iter().enumerate() {
        let src_node = &graph.nodes[graph.source_indices[i]];
        let fired = firing[i];
        for te in src_node.trigger_edges.iter() {
            let down = &graph.nodes[te.downstream];
            let down_bk = down.bookkeeping.as_ref().expect(
                "downstream of a source (a wavefront participant) must have bookkeeping",
            );
            if fired {
                down_bk.set_bit(tick_slot, te.input_pos);
            }
            let prev = down_bk.dec_remaining(tick_slot);
            if prev == 1 {
                state.enqueue(Task {
                    node_idx: te.downstream,
                    tick_no,
                    ts,
                });
            }
        }
    }

    // Operators with `effective_upstream_count == 0` (not Const — actually
    // those have arity == 0 already and participates_in_wavefront == false
    // so no trigger edges; any operator with upstreams that are all
    // Const would fall here) should fire immediately without waiting for
    // any signal.  Iterate and enqueue them.
    //
    // Note: a wavefront-participant operator always has arity > 0; if all
    // its upstreams are Const, effective_upstream_count == 0 — it has
    // no signal sources, so it must be kicked off here.  In practice,
    // this is rare but we handle it for completeness.
    for node in graph.nodes.iter() {
        if node.kind == NodeKind::Operator
            && node.participates_in_wavefront
            && node.effective_upstream_count == 0
        {
            state.enqueue(Task {
                node_idx: node.index,
                tick_no,
                ts,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Commit-barrier wait
// ---------------------------------------------------------------------------

/// Block until every operator node's `last_committed_tick >= required`.
/// Called by ingest to respect the pipeline-width invariant: no more than
/// W ticks in flight at any time.
async fn wait_for_commit(graph: &Graph, required: i64) {
    loop {
        let ok = graph.nodes.iter().all(|n| {
            // Only operator nodes that participate in the wavefront need
            // to commit per-tick.  Sources commit only on event ingest,
            // so their last_committed_tick doesn't track the global tick
            // count — we exclude them.
            !n.participates_in_wavefront
                || n.kind == NodeKind::Source
                || n.last_committed_tick.load(Ordering::Acquire) >= required
        });
        if ok {
            return;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
