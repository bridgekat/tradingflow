//! Worker pool and task dispatch for the wavefront runtime.
//!
//! The scheduler is driven by a [`crossbeam_deque::Injector`] fed by both
//! the ingest loop (for bootstrap-ready nodes such as "all-upstreams
//! signalled" cases at each tick) and by operator tasks completing (which
//! decrement their downstreams' remaining-inputs counters and re-push the
//! downstream once it hits zero).
//!
//! # Readiness predicate
//!
//! A task `(node, tick_no, ts)` is dispatched once its node's
//! `remaining_inputs[tick_no mod W]` reaches zero.  At the moment a
//! worker picks it up it then verifies the **temporal self-edge**:
//! `node.last_committed_tick >= tick_no - 1`.  If not satisfied (the
//! previous tick's compute for this node has not yet finalised), the
//! worker re-injects the task and continues — the node's same-tick
//! serialisation is implicit via `last_committed_tick`, and
//! re-injection cheaply backs off until the predecessor commits.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crossbeam_deque::{Injector, Steal, Stealer, Worker};
use crossbeam_utils::Backoff;

use super::super::data::Instant;
use super::graph::Graph;
use super::node::BorrowGuard;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Task dispatched to a worker: "run compute for `node_idx` at `ts`
/// (logical tick `tick_no`)."
#[derive(Debug, Copy, Clone)]
pub struct Task {
    pub node_idx: usize,
    pub tick_no: u64,
    pub ts: Instant,
}

/// Cooperative shutdown flag (shared `AtomicBool`).
#[derive(Clone)]
pub struct ShutdownFlag(pub Arc<AtomicBool>);

impl ShutdownFlag {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
    pub fn trigger(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

impl Default for ShutdownFlag {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared between the ingest loop and all worker threads.
pub struct SharedState {
    pub graph: Arc<Graph>,
    pub injector: Injector<Task>,
    pub stealers: Vec<Stealer<Task>>,
    /// Set once all sources have closed and ingest has finished.
    pub ingest_done: AtomicBool,
    /// Cooperative shutdown — bail out of the compute loop ASAP.
    pub shutdown: Arc<AtomicBool>,
    /// Total outstanding tasks (enqueued but not yet executed).
    pub pending_tasks: AtomicUsize,
    /// Waker for parked workers and the drain waiter.
    pub park_mutex: Mutex<()>,
    pub park_cv: Condvar,
    pub pipeline_width: usize,
}

impl SharedState {
    /// Push a task to the global injector and wake one worker.
    pub fn enqueue(&self, task: Task) {
        self.pending_tasks.fetch_add(1, Ordering::AcqRel);
        self.injector.push(task);
        self.park_cv.notify_one();
    }

    /// Wake every parked worker (e.g., after shutdown flag flipped).
    pub fn notify_all(&self) {
        self.park_cv.notify_all();
    }

    fn decrement_pending(&self) {
        let prev = self.pending_tasks.fetch_sub(1, Ordering::AcqRel);
        if prev <= 1 {
            self.park_cv.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// Drive entry point
// ---------------------------------------------------------------------------

/// Run the ingest + scheduler to completion.
///
/// Must be called from an async context so the ingest-side tokio mpsc
/// channels can be polled.  Workers run on `std::thread` and are joined
/// before this function returns.
pub async fn drive(graph: Arc<Graph>, shutdown: ShutdownFlag) {
    let pipeline_width = graph.pipeline_width();
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2);

    // Build per-worker deques + stealers, then the shared state.
    let worker_deques: Vec<Worker<Task>> = (0..n_workers).map(|_| Worker::new_fifo()).collect();
    let stealers: Vec<Stealer<Task>> = worker_deques.iter().map(|w| w.stealer()).collect();

    let state = Arc::new(SharedState {
        graph: Arc::clone(&graph),
        injector: Injector::<Task>::new(),
        stealers,
        ingest_done: AtomicBool::new(false),
        shutdown: Arc::clone(&shutdown.0),
        pending_tasks: AtomicUsize::new(0),
        park_mutex: Mutex::new(()),
        park_cv: Condvar::new(),
        pipeline_width,
    });

    // Spawn workers.
    let mut worker_handles = Vec::with_capacity(n_workers);
    for (i, w) in worker_deques.into_iter().enumerate() {
        let state_w = Arc::clone(&state);
        worker_handles.push(std::thread::spawn(move || worker_loop(state_w, w, i)));
    }

    // Drive ingest on this task.
    super::ingest::ingest_main(&state).await;
    state.ingest_done.store(true, Ordering::Release);
    state.notify_all();

    // Wait for all dispatched tasks to drain.
    {
        let mut g = state.park_mutex.lock().unwrap();
        loop {
            if state.pending_tasks.load(Ordering::Acquire) == 0 {
                break;
            }
            if state.shutdown.load(Ordering::Acquire) {
                break;
            }
            g = state.park_cv.wait_timeout(g, Duration::from_millis(50)).unwrap().0;
        }
    }

    // Signal shutdown and wake all workers.
    state.shutdown.store(true, Ordering::Release);
    state.notify_all();

    for h in worker_handles {
        let _ = h.join();
    }

    // Close all output queues to release any parked pushers (should be
    // none by now, but defensive).
    for node in graph.nodes.iter() {
        node.output.close();
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(state: Arc<SharedState>, worker: Worker<Task>, _worker_id: usize) {
    let backoff = Backoff::new();
    loop {
        if let Some(task) = find_task(&state, &worker) {
            execute_task(&state, task);
            backoff.reset();
        } else {
            let done = state.ingest_done.load(Ordering::Acquire)
                && state.pending_tasks.load(Ordering::Acquire) == 0;
            let kill = state.shutdown.load(Ordering::Acquire);
            if done || kill {
                break;
            }
            if backoff.is_completed() {
                let g = state.park_mutex.lock().unwrap();
                // Recheck under lock.
                if state.pending_tasks.load(Ordering::Acquire) == 0
                    && !state.ingest_done.load(Ordering::Acquire)
                    && !state.shutdown.load(Ordering::Acquire)
                {
                    let _ = state
                        .park_cv
                        .wait_timeout(g, Duration::from_millis(5))
                        .unwrap();
                }
                backoff.reset();
            } else {
                backoff.snooze();
            }
        }
    }
}

fn find_task(state: &SharedState, worker: &Worker<Task>) -> Option<Task> {
    if let Some(t) = worker.pop() {
        return Some(t);
    }
    loop {
        match state.injector.steal_batch_and_pop(worker) {
            Steal::Success(t) => return Some(t),
            Steal::Empty => break,
            Steal::Retry => continue,
        }
    }
    for stealer in state.stealers.iter() {
        loop {
            match stealer.steal_batch_and_pop(worker) {
                Steal::Success(t) => return Some(t),
                Steal::Empty => break,
                Steal::Retry => continue,
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Task execution
// ---------------------------------------------------------------------------

fn execute_task(state: &SharedState, task: Task) {
    let graph = &state.graph;
    let node = &graph.nodes[task.node_idx];

    // Temporal self-edge readiness.  If the prior tick hasn't committed
    // yet, re-inject this task and let another work item proceed.  Same-
    // node tasks are implicitly serialised through this check.
    let expected_prev = task.tick_no as i64 - 1;
    if node.last_committed_tick.load(Ordering::Acquire) < expected_prev {
        // Re-enqueue (does not change pending_tasks).
        state.injector.push(task);
        std::thread::yield_now();
        return;
    }

    let width = state.pipeline_width;
    let tick_slot = (task.tick_no as usize) % width;

    // Collect input pointers and hold their underlying `Arc<Slot<T>>`
    // clones alive via BorrowGuards for the lifetime of compute.
    let mut input_ptrs: Vec<*const u8> = Vec::with_capacity(node.input_edges.len());
    let mut borrow_guards: Vec<BorrowGuard> = Vec::with_capacity(node.input_edges.len());
    for (upstream_idx, _pos) in node.input_edges.iter() {
        let up = &graph.nodes[*upstream_idx];
        let (p, g) = up.output.borrow(task.ts).unwrap_or_else(|| {
            panic!(
                "upstream node {} has no committed slot at ts {:?} (tick {})",
                upstream_idx, task.ts, task.tick_no,
            )
        });
        input_ptrs.push(p.0);
        borrow_guards.push(g);
    }

    // Snapshot the produced-bits for this tick before running compute.
    let bk = node
        .bookkeeping
        .as_ref()
        .expect("operator with inputs must have bookkeeping");
    let bits = bk.snapshot_bits(tick_slot);
    let any_fired = bits.iter().any(|w| *w != 0);

    let did_produce = if any_fired {
        let output_ptr = node.output.alloc_fresh();
        let compute_fn = node
            .compute_fn
            .expect("operator node must have compute_fn");
        let ok = unsafe {
            compute_fn(
                node.state_ptr.0,
                &input_ptrs,
                output_ptr,
                task.ts,
                &bits,
                0,
                node.arity,
            )
        };
        if ok {
            unsafe {
                node.output.commit(task.ts, output_ptr);
            }
        } else {
            unsafe {
                node.output.drop_fresh(output_ptr);
            }
        }
        ok
    } else {
        false
    };

    // Clear bits for reuse when this tick-slot wraps around.  Safe because
    // the slot isn't reassigned to a new tick until all downstream
    // signals for this tick have been dispatched — enforced by pipeline
    // width W and the per-tick reset done by ingest.
    for w in bk.incoming_bits[tick_slot].iter() {
        w.store(0, Ordering::Release);
    }

    // Advance last_committed_tick (enables same-node t+1 to proceed).
    node.last_committed_tick
        .store(task.tick_no as i64, Ordering::Release);
    node.last_committed_ts
        .store(task.ts.as_nanos(), Ordering::Release);

    // Retire old slots in the output queue.
    node.output.retire();

    // Drop input borrow guards.
    drop(borrow_guards);

    // Propagate signals to downstreams.
    for te in node.trigger_edges.iter() {
        let down = &graph.nodes[te.downstream];
        let down_bk = down.bookkeeping.as_ref().expect(
            "downstream of a wavefront participant must have bookkeeping (has upstream edges)",
        );
        if did_produce {
            down_bk.set_bit(tick_slot, te.input_pos);
        }
        let prev = down_bk.dec_remaining(tick_slot);
        if prev == 1 {
            state.enqueue(Task {
                node_idx: te.downstream,
                tick_no: task.tick_no,
                ts: task.ts,
            });
        }
    }

    state.decrement_pending();
}
