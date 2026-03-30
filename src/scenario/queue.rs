//! Async source handling and the POCQ event loop.
//!
//! [`SourceState`](super::node::SourceState) tracks per-source runtime state
//! inside each source [`Node`](super::node::Node).  The [`Scenario::run`]
//! method implements the Point-of-Coherency Queue (POCQ) algorithm that
//! consumes all registered sources and propagates events through the DAG.
//!
//! # Algorithm
//!
//! The event loop maintains a min-heap (`BinaryHeap<Reverse<HeapEntry>>`) of
//! pending `(timestamp, source_idx, kind)` triples, alongside two
//! `FuturesUnordered<ErasedRecvFuture>` collections — one for historical
//! re-fills (`hist_refills`) and one for live re-fills (`live_refills`).
//! This gives **O(log N) per-event** cost for N sources.
//!
//! **Historical constraint**: before any pop, every non-exhausted historical
//! channel must have its next event in the heap.  Enforced by `drain_hist`,
//! which blocks concurrently on all in-flight historical re-fills but may
//! exit early once it is provably safe to do so:
//!
//! Because historical channels are non-decreasing, the next event from source
//! `i` has timestamp ≥ `last_hist_ts[i]` (the last consumed timestamp).  A
//! `BTreeMap` multiset (`hist_pending_ts`) tracks the minimum such lower bound
//! across all in-flight futures in O(log N).  Once that minimum is strictly
//! greater than the current heap minimum, no pending future can produce an
//! event that beats the heap minimum, so it is safe to pop without waiting.
//!
//! **Live channels**: managed symmetrically via `live_refills`.  `drain_live`
//! is a non-blocking poll — it drains all currently-ready live futures without
//! suspending the task.  When the heap runs dry, the loop blocks on
//! `live_refills.next()` until the next live event arrives.
//!
//! **Timestamp clamping**: live events are inserted into the heap with
//! `ts = ts.max(current_ts)` so that ingested timestamps are non-decreasing.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::future::poll_fn;
use std::pin::Pin;
use std::task::Poll;

use futures::stream::{FuturesUnordered, StreamExt};

use crate::source::PollFn;

use super::Scenario;
use super::node::{ChannelKind, SourceState};

// ---------------------------------------------------------------------------
// recv_future — queue-specific extension on SourceState
// ---------------------------------------------------------------------------

impl SourceState {
    /// Create a future that resolves to `(source_idx, kind, Option<i64>)`.
    ///
    /// # Safety
    ///
    /// The returned [`ErasedRecvFuture`] must not outlive `self`.
    pub(super) unsafe fn recv_future(
        &self,
        source_idx: usize,
        kind: ChannelKind,
    ) -> ErasedRecvFuture {
        ErasedRecvFuture::new(self.rx_ptr(kind), self.poll_fn(), source_idx, kind)
    }
}

// ---------------------------------------------------------------------------
// ErasedRecvFuture — one pending channel receive
// ---------------------------------------------------------------------------

/// A `'static + Send` future representing one pending channel receive.
///
/// Resolves to `(source_idx, kind, Option<i64>)` — `None` means the channel
/// closed.  Designed for use in [`FuturesUnordered`] so the queue can
/// concurrently await many channels with O(log N) heap inserts.
///
/// # Safety invariant
///
/// Must not outlive the [`SourceState`] it was created from.
pub struct ErasedRecvFuture {
    rx_ptr: *mut u8,
    poll_fn: PollFn,
    source_idx: usize,
    kind: ChannelKind,
}

// SAFETY: rx_ptr points to PeekableReceiver<E>: Send.
unsafe impl Send for ErasedRecvFuture {}

impl ErasedRecvFuture {
    fn new(rx_ptr: *mut u8, poll_fn: PollFn, source_idx: usize, kind: ChannelKind) -> Self {
        Self {
            rx_ptr,
            poll_fn,
            source_idx,
            kind,
        }
    }
}

impl std::future::Future for ErasedRecvFuture {
    /// `(source_idx, kind, Option<i64>)` — `None` timestamp means channel closed.
    type Output = (usize, ChannelKind, Option<i64>);

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let source_idx = self.source_idx;
        let kind = self.kind;
        unsafe { (self.poll_fn)(self.rx_ptr, cx) }.map(|opt_ts| (source_idx, kind, opt_ts))
    }
}

// ---------------------------------------------------------------------------
// HeapEntry — one timestamped event in the merge heap
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
struct HeapEntry {
    ts: i64,
    source_idx: usize,
    kind: ChannelKind,
}

// ---------------------------------------------------------------------------
// drain_hist — blocking drain with early-exit
// ---------------------------------------------------------------------------

/// Drain `hist_refills` until the historical constraint is satisfied.
///
/// Blocks concurrently on all in-flight historical re-fills but exits early
/// once it is provably safe: when the minimum lower bound of all still-pending
/// futures (`hist_pending_ts.keys().next()`) is strictly greater than the
/// current heap minimum, no pending future can produce an event that beats
/// the heap minimum.
///
/// `last_hist_ts[i]` holds the timestamp of the last consumed hist event for
/// source `i` (a valid non-decreasing lower bound on its next event).
/// `hist_pending_ts` is a `BTreeMap` multiset of those lower-bound values for
/// all sources whose futures are currently in `hist_refills`.
async fn drain_hist(
    hist_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    last_hist_ts: &mut [i64],
    hist_pending_ts: &mut BTreeMap<i64, usize>,
) {
    while let Some((src, _kind, opt_ts)) = hist_refills.next().await {
        // Remove this source's lower-bound entry from the pending multiset.
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
        // If opt_ts is None the hist channel is closed; no re-queue.

        if hist_refills.is_empty() {
            break;
        }

        // Early-exit: if all remaining pending futures are guaranteed to
        // produce timestamps strictly greater than the current heap minimum,
        // we can safely pop the heap minimum without waiting for them.
        let heap_min = heap.peek().map(|&Reverse(e)| e.ts).unwrap_or(i64::MAX);
        let pending_lower = hist_pending_ts.keys().next().copied().unwrap_or(i64::MAX);
        if pending_lower > heap_min {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// drain_live — non-blocking drain of live_refills
// ---------------------------------------------------------------------------

/// Drain all currently-ready futures from `live_refills` without blocking.
///
/// Uses `poll_fn` to poll the `FuturesUnordered` once per iteration: drains
/// every future that is already ready, then returns immediately when the
/// collection is empty or when no further futures are ready right now.
/// Registers the task waker so the task is re-woken when new live events
/// arrive.
///
/// Live events are inserted into the heap with `ts = ts.max(current_ts)` to
/// maintain non-decreasing ingestion order.
async fn drain_live(
    live_refills: &mut FuturesUnordered<ErasedRecvFuture>,
    heap: &mut BinaryHeap<Reverse<HeapEntry>>,
    current_ts: i64,
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
                    // Live channel closed; source exhausted — no re-queue.
                }
                Poll::Ready(None) | Poll::Pending => return Poll::Ready(()),
            }
        }
    })
    .await;
}

// ---------------------------------------------------------------------------
// Scenario — POCQ event loop
// ---------------------------------------------------------------------------

impl Scenario {
    /// Run the unified POCQ event loop.
    ///
    /// Consumes all historical and live events from every registered source
    /// in timestamp order, propagating each batch through the DAG via
    /// [`Graph::flush`].
    ///
    /// # Ordering guarantees
    ///
    /// * **Historical constraint**: before ingesting the event with the
    ///   globally smallest timestamp, every active historical channel has its
    ///   next event available in the heap.
    /// * **Non-decreasing timestamps**: live events with timestamps earlier
    ///   than the last ingested batch timestamp are clamped to that timestamp.
    ///
    /// # Complexity
    ///
    /// O(log N) per event for N sources — no O(N) scans.
    pub async fn run(&mut self) {
        let n = self.source_indices.len();
        if n == 0 {
            return;
        }

        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        let mut hist_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
        let mut live_refills: FuturesUnordered<ErasedRecvFuture> = FuturesUnordered::new();
        let mut current_ts: i64 = i64::MIN;

        // Per-source lower bounds for the drain_hist early-exit check.
        // last_hist_ts[i] = timestamp of last consumed hist event for source i
        //                   (valid lower bound on its next event, i64::MIN initially).
        let mut last_hist_ts: Vec<i64> = vec![i64::MIN; n];
        // BTreeMap multiset: lower-bound value → count of pending futures with that bound.
        let mut hist_pending_ts: BTreeMap<i64, usize> = BTreeMap::new();

        // ----------------------------------------------------------------
        // Initialisation: push all hist and live futures.
        // ----------------------------------------------------------------
        for i in 0..n {
            // SAFETY: source state inside graph.nodes outlives all futures
            // created here; both live for the entire duration of run().
            let source = self.graph.nodes[self.source_indices[i]]
                .source_state()
                .unwrap();
            hist_refills.push(unsafe { source.recv_future(i, ChannelKind::Hist) });
            *hist_pending_ts.entry(i64::MIN).or_insert(0) += 1;
            live_refills.push(unsafe { source.recv_future(i, ChannelKind::Live) });
        }

        // Drain all historical first events (blocking, concurrent).
        // Early-exit cannot fire during init: all lower bounds are i64::MIN.
        drain_hist(
            &mut hist_refills,
            &mut heap,
            &mut last_hist_ts,
            &mut hist_pending_ts,
        )
        .await;

        // Non-blocking initial poll of live channels.
        drain_live(&mut live_refills, &mut heap, current_ts).await;

        let mut queue_ts: Option<i64> = None;
        let mut queue_sources: Vec<usize> = Vec::new();

        loop {
            // ----------------------------------------------------------------
            // Step 1: Historical constraint.
            // Drain hist_refills until all pending futures are safe to skip.
            // ----------------------------------------------------------------
            drain_hist(
                &mut hist_refills,
                &mut heap,
                &mut last_hist_ts,
                &mut hist_pending_ts,
            )
            .await;

            // ----------------------------------------------------------------
            // Step 2: Non-blocking drain of live channels.
            // ----------------------------------------------------------------
            drain_live(&mut live_refills, &mut heap, current_ts).await;

            // ----------------------------------------------------------------
            // Step 3: If heap is empty, either finish or block for a live event.
            // ----------------------------------------------------------------
            let Some(&Reverse(HeapEntry { ts: min_ts, .. })) = heap.peek() else {
                if live_refills.is_empty() {
                    break;
                }
                // Block until any live channel fires.
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

            // ----------------------------------------------------------------
            // Step 4: Coalesce — flush accumulated batch when ts advances.
            // ----------------------------------------------------------------
            if let Some(qts) = queue_ts
                && min_ts > qts
            {
                self.graph.flush(qts, &queue_sources);
                queue_sources.clear();
            }

            // ----------------------------------------------------------------
            // Step 5: Pop and process every entry at min_ts.
            // ----------------------------------------------------------------
            while let Some(&Reverse(HeapEntry { ts, .. })) = heap.peek() {
                if ts > min_ts {
                    break;
                }
                let Reverse(HeapEntry {
                    ts: _,
                    source_idx,
                    kind,
                }) = heap.pop().unwrap();

                let node_idx = self.source_indices[source_idx];
                let node = &self.graph.nodes[node_idx];
                let source = node.source_state().unwrap();
                let rx_ptr = source.rx_ptr(kind);
                let write_fn = source.write_fn();
                unsafe { (write_fn)(rx_ptr, node.value_ptr, min_ts) };
                queue_sources.push(node_idx);

                // Re-queue the consumed channel's future.
                let source = self.graph.nodes[node_idx].source_state().unwrap();
                match kind {
                    ChannelKind::Hist => {
                        // Register the new lower bound for this source's next
                        // future.  drain_hist already removed the previous entry
                        // for this source when it resolved the future, so we
                        // only need to add the new one.
                        last_hist_ts[source_idx] = min_ts;
                        *hist_pending_ts.entry(min_ts).or_insert(0) += 1;
                        hist_refills
                            .push(unsafe { source.recv_future(source_idx, ChannelKind::Hist) });
                    }
                    ChannelKind::Live => {
                        live_refills
                            .push(unsafe { source.recv_future(source_idx, ChannelKind::Live) });
                    }
                }
            }
            current_ts = min_ts;
            queue_ts = Some(min_ts);
        }

        // Final flush for the last accumulated batch.
        if !queue_sources.is_empty() {
            self.graph.flush(queue_ts.unwrap(), &queue_sources);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — randomized POCQ invariant checks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::{Arc, Mutex};

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tokio::sync::mpsc;

    use crate::Scenario;
    use crate::array::Array;
    use crate::operator::Operator;
    use crate::operators::Record;
    use crate::series::Series;
    use crate::source::Source;

    use super::super::handle::Handle;

    // -- Test source ----------------------------------------------------------

    /// A source backed by pre-filled bounded channels.
    struct PrefilledSource {
        hist_events: Vec<(i64, f64)>,
        live_events: Vec<(i64, f64)>,
    }

    impl Source for PrefilledSource {
        type Event = f64;
        type Output = Array<f64>;

        fn init(
            self,
            _ts: i64,
        ) -> (
            mpsc::Receiver<(i64, f64)>,
            mpsc::Receiver<(i64, f64)>,
            Array<f64>,
        ) {
            let (hist_tx, hist_rx) = mpsc::channel(self.hist_events.len().max(1));
            for evt in &self.hist_events {
                hist_tx.try_send(*evt).unwrap();
            }
            drop(hist_tx);

            let (live_tx, live_rx) = mpsc::channel(self.live_events.len().max(1));
            for evt in &self.live_events {
                live_tx.try_send(*evt).unwrap();
            }
            drop(live_tx);

            (hist_rx, live_rx, Array::scalar(0.0_f64))
        }

        fn write(event: f64, output: &mut Array<f64>, _ts: i64) -> bool {
            output[0] = event;
            true
        }
    }

    fn make_source(hist: &[(i64, f64)], live: &[(i64, f64)]) -> PrefilledSource {
        PrefilledSource {
            hist_events: hist.to_vec(),
            live_events: live.to_vec(),
        }
    }

    // -- Global logger operator -----------------------------------------------

    struct GlobalLogger {
        source_id: usize,
        log: Arc<Mutex<Vec<(i64, usize)>>>,
    }

    impl Operator for GlobalLogger {
        type State = (usize, Arc<Mutex<Vec<(i64, usize)>>>);
        type Inputs = (Array<f64>,);
        type Output = ();

        fn init(
            self,
            _inputs: (&Array<f64>,),
            _ts: i64,
        ) -> ((usize, Arc<Mutex<Vec<(i64, usize)>>>), ()) {
            ((self.source_id, self.log), ())
        }

        fn compute(
            state: &mut (usize, Arc<Mutex<Vec<(i64, usize)>>>),
            _inputs: (&Array<f64>,),
            _output: &mut (),
            timestamp: i64,
        ) -> bool {
            state.1.lock().unwrap().push((timestamp, state.0));
            false
        }
    }

    // -- Helpers ---------------------------------------------------------------

    fn generate_events(rng: &mut StdRng, source_id: usize) -> (Vec<(i64, f64)>, Vec<(i64, f64)>) {
        let hist_count: usize = rng.gen_range(0..=15);
        let live_count: usize = rng.gen_range(0..=15);

        let mut hist_ts: Vec<i64> = (0..hist_count).map(|_| rng.gen_range(0..=100)).collect();
        hist_ts.sort();

        let hist_max = hist_ts.last().copied().unwrap_or(0);
        let mut live_ts: Vec<i64> = (0..live_count)
            .map(|_| rng.gen_range(hist_max..=200))
            .collect();
        live_ts.sort();

        let hist: Vec<(i64, f64)> = hist_ts
            .into_iter()
            .enumerate()
            .map(|(i, ts)| (ts, source_id as f64 * 10000.0 + i as f64))
            .collect();

        let live: Vec<(i64, f64)> = live_ts
            .into_iter()
            .enumerate()
            .map(|(i, ts)| (ts, source_id as f64 * 10000.0 + 1000.0 + i as f64))
            .collect();

        (hist, live)
    }

    // -- Invariant checker ----------------------------------------------------

    /// Check the POCQ output invariants for one source.
    ///
    /// 1. Monotonicity — recorded timestamps are non-decreasing.
    /// 2. Historical stability — every distinct hist timestamp appears
    ///    at its original value.
    /// 3. Live delay-only — entries after the hist block have timestamps
    ///    ≥ the minimum original live timestamp.
    /// 4. Completeness — at least as many entries as distinct hist timestamps.
    /// 5. No fabrication — no recorded timestamp below the minimum original.
    fn check_invariants(
        series: &Series<f64>,
        hist_events: &[(i64, f64)],
        live_events: &[(i64, f64)],
        source_id: usize,
        seed: u64,
    ) {
        let ts = series.timestamps();

        for w in ts.windows(2) {
            assert!(
                w[0] <= w[1],
                "seed {seed}, source {source_id}: monotonicity violated: {} > {}",
                w[0],
                w[1],
            );
        }

        let distinct_hist: BTreeSet<i64> = hist_events.iter().map(|e| e.0).collect();
        let recorded: BTreeSet<i64> = ts.iter().copied().collect();
        for &ht in &distinct_hist {
            assert!(
                recorded.contains(&ht),
                "seed {seed}, source {source_id}: hist timestamp {ht} missing",
            );
        }
        let hist_expected: Vec<i64> = distinct_hist.iter().copied().collect();
        if ts.len() >= hist_expected.len() {
            assert_eq!(
                &ts[..hist_expected.len()],
                &hist_expected[..],
                "seed {seed}, source {source_id}: hist portion of record is wrong",
            );
        }

        if !live_events.is_empty() {
            let min_live_ts = live_events[0].0;
            for (i, &t) in ts.iter().enumerate().skip(hist_expected.len()) {
                assert!(
                    t >= min_live_ts,
                    "seed {seed}, source {source_id}: live entry {i} ts={t} < min_live_ts={min_live_ts}",
                );
            }
        }

        assert!(
            ts.len() >= distinct_hist.len(),
            "seed {seed}, source {source_id}: too few entries: {} < {} distinct hist",
            ts.len(),
            distinct_hist.len(),
        );

        if !ts.is_empty() {
            let min_original = hist_events
                .iter()
                .chain(live_events.iter())
                .map(|e| e.0)
                .min()
                .unwrap();
            assert!(
                ts[0] >= min_original,
                "seed {seed}, source {source_id}: first ts {} < min original {}",
                ts[0],
                min_original,
            );
        }
    }

    fn check_global_log(log: &[(i64, usize)], seed: u64) {
        for w in log.windows(2) {
            assert!(
                w[0].0 <= w[1].0,
                "seed {seed}: global monotonicity violated: ts {} (source {}) > ts {} (source {})",
                w[0].0,
                w[0].1,
                w[1].0,
                w[1].1,
            );
        }
    }

    // -- Tests ----------------------------------------------------------------

    /// Zero sources — must terminate immediately.
    #[tokio::test]
    async fn pocq_empty_scenario() {
        let mut sc = Scenario::new();
        sc.run().await;
    }

    /// A stale live event (ts=50) arriving after hist pushes current_ts to 100
    /// must be clamped to current_ts and coalesced, not ingested at ts=50.
    ///
    /// Uses `tokio::spawn` to send the live event after run() suspends waiting
    /// for live input, deterministically exercising the `ts.max(current_ts)`
    /// clamping path.
    #[tokio::test]
    async fn pocq_live_clamping() {
        use crate::operators::Record;

        struct ManualChannel {
            hist_rx: mpsc::Receiver<(i64, f64)>,
            live_rx: mpsc::Receiver<(i64, f64)>,
        }

        impl Source for ManualChannel {
            type Event = f64;
            type Output = Array<f64>;

            fn init(
                self,
                _timestamp: i64,
            ) -> (
                mpsc::Receiver<(i64, f64)>,
                mpsc::Receiver<(i64, f64)>,
                Array<f64>,
            ) {
                (self.hist_rx, self.live_rx, Array::scalar(0.0_f64))
            }

            fn write(event: f64, output: &mut Array<f64>, _timestamp: i64) -> bool {
                output[0] = event;
                true
            }
        }

        let (hist_tx, hist_rx) = mpsc::channel(1);
        let (live_tx, live_rx) = mpsc::channel(1);
        hist_tx.send((100_i64, 1.0_f64)).await.unwrap();
        drop(hist_tx);

        let mut sc = Scenario::new();
        let hs = sc.add_source(ManualChannel { hist_rx, live_rx });
        let hrec = sc.add_operator(Record::<f64>::new(), (hs,), None);

        tokio::spawn(async move {
            live_tx.send((50_i64, 2.0_f64)).await.unwrap();
        });

        sc.run().await;

        let series: &Series<f64> = sc.value(hrec);
        assert_eq!(series.len(), 1);
        assert_eq!(series.timestamps(), &[100_i64]);
    }

    /// 200 random scenarios testing interleaving of hist and live channels
    /// across multiple sources.
    #[tokio::test]
    async fn pocq_random_interleaving() {
        for seed in 0..200 {
            let mut rng = StdRng::seed_from_u64(seed);
            let n_sources: usize = rng.gen_range(1..=5);

            let mut sc = Scenario::new();
            let log = Arc::new(Mutex::new(Vec::new()));
            let mut source_data: Vec<(Vec<(i64, f64)>, Vec<(i64, f64)>)> = Vec::new();
            let mut records: Vec<Handle<Series<f64>>> = Vec::new();

            for i in 0..n_sources {
                let (hist, live) = generate_events(&mut rng, i);
                let h = sc.add_source(make_source(&hist, &live));
                records.push(sc.add_operator(Record::<f64>::new(), (h,), None));
                sc.add_operator(
                    GlobalLogger {
                        source_id: i,
                        log: log.clone(),
                    },
                    (h,),
                    None,
                );
                source_data.push((hist, live));
            }

            sc.run().await;

            for (i, (hr, (hist, live))) in records.iter().zip(source_data.iter()).enumerate() {
                let series: &Series<f64> = sc.value(*hr);
                check_invariants(series, hist, live, i, seed);
            }
            check_global_log(&log.lock().unwrap(), seed);
        }
    }
}
