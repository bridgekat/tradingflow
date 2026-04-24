//! Graph node — type-erased with per-tick bookkeeping atomics.
//!
//! A [`Node`] is one vertex of the wavefront computation graph.  It owns:
//!
//! * heap-allocated operator state (`state_ptr: *mut u8`) and an output
//!   template used to mint fresh per-timestamp output buffers.
//! * a type-erased [`OutputStore`] trait object that manages the actual
//!   committed values (auto-GC queue for the PoC).
//! * per-tick readiness atomics indexed by `tick_no mod W`: the count of
//!   upstream edges that have yet to signal, and a packed bitset of which
//!   input positions have fired.
//! * a [`last_committed_tick`](Node::last_committed_tick) atomic that
//!   enforces the temporal self-edge (compute at tick `k+1` may not begin
//!   before compute at `k` commits).
//! * a [`compute_mutex`](Node::compute_mutex) serialising same-node
//!   computes.  Correctness alone doesn't require this (the self-edge
//!   already serialises same-node ticks), but it gives `compute` a cheap
//!   `&mut State` guarantee without `unsafe` wrapping.

use std::any::TypeId;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::super::data::Instant;
use super::super::operator::{CloneFn, ComputeFn, DropFn, NodeKind};
use super::super::queue::{OutputQueue, Slot};

// ===========================================================================
// OutputStore trait
// ===========================================================================

/// Type-erased handle to a node's output storage.
///
/// Implementations are typically backed by an [`OutputQueue<T>`] holding
/// `Arc<Slot<T>>` entries; see [`QueueStore`].
pub trait OutputStore: Send + Sync + 'static {
    /// [`TypeId`] of the underlying `T`.
    fn output_type_id(&self) -> TypeId;

    /// Borrow the latest committed slot whose `ts <= requested_ts`, if any.
    ///
    /// Returns a raw pointer to the value and a [`BorrowGuard`] that must
    /// be held for the lifetime of the borrow (keeps the underlying
    /// `Arc<Slot<T>>` alive).
    fn borrow(&self, requested_ts: Instant) -> Option<(OutputPtr, BorrowGuard)>;

    /// Commit a freshly-computed value at `ts`.  Takes ownership of the
    /// boxed value behind `value_box_ptr` (a `Box<Output>` from
    /// [`alloc_fresh`](Self::alloc_fresh)).
    ///
    /// # Safety
    ///
    /// `value_box_ptr` must be a `Box<T>` for the same `T` as
    /// [`output_type_id`](Self::output_type_id), and ownership is
    /// transferred here.
    unsafe fn commit(&self, ts: Instant, value_box_ptr: *mut u8) -> bool;

    /// Allocate a fresh output buffer by cloning the init-template.  The
    /// returned pointer is a `Box<T>::into_raw()`; the caller passes it to
    /// [`commit`](Self::commit) on success or [`drop_fresh`](Self::drop_fresh)
    /// on abandonment.
    fn alloc_fresh(&self) -> *mut u8;

    /// Drop a fresh buffer that will not be committed.
    ///
    /// # Safety
    ///
    /// `ptr` must have been returned by [`alloc_fresh`](Self::alloc_fresh)
    /// on this same store.
    unsafe fn drop_fresh(&self, ptr: *mut u8);

    /// Retire any front slots whose only remaining `Arc` holder is the
    /// queue.  Called after each commit.
    fn retire(&self);

    /// Close the store — parked writers return `false` from commit.
    fn close(&self);
}

/// Raw pointer wrapper explicitly `Send + Sync`.
///
/// The scheduler carries these across worker threads; the underlying
/// pointer is valid for the lifetime of the accompanying [`BorrowGuard`]
/// because the store's slot (`Arc<Slot<T>>`) is pinned while the guard
/// holds a clone.
#[derive(Copy, Clone)]
pub struct OutputPtr(pub *const u8);

unsafe impl Send for OutputPtr {}
unsafe impl Sync for OutputPtr {}

/// Opaque holder that keeps an output's backing allocation alive while a
/// reader is using the raw pointer.  Drop to release.
pub struct BorrowGuard(Option<Arc<dyn std::any::Any + Send + Sync>>);

impl BorrowGuard {
    /// Empty guard.  Used by storages that don't need lifetime pinning.
    pub fn none() -> Self {
        Self(None)
    }

    /// Guard that holds an arbitrary `Arc` alive.
    pub fn from_arc<S: std::any::Any + Send + Sync>(a: Arc<S>) -> Self {
        let erased: Arc<dyn std::any::Any + Send + Sync> = a;
        Self(Some(erased))
    }

    /// Consume the guard and return the inner `Arc<dyn Any>`, if any.
    /// Used for downcasting back to `Arc<Slot<T>>` at inspection time.
    pub fn into_any(self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.0
    }
}

// ===========================================================================
// QueueStore — default OutputStore backed by OutputQueue<T>
// ===========================================================================

/// [`OutputStore`] backed by an [`OutputQueue<T>`].
///
/// This is the default storage for all output types in the PoC.  Each
/// call to [`alloc_fresh`](Self::alloc_fresh) clones the **latest
/// committed slot** (falling back to the seed template when nothing has
/// committed yet) so stateful outputs like [`Series<T>`](crate::Series)
/// observe the accumulated prior state instead of being reset each tick.
/// For replacement outputs (e.g. `Array<T>`) the extra clone is
/// equivalent to cloning the template — `compute` overwrites the
/// contents anyway.
pub struct QueueStore<T: Clone + Send + Sync + 'static> {
    queue: Arc<OutputQueue<T>>,
}

impl<T: Clone + Send + Sync + 'static> QueueStore<T> {
    pub fn new(template: T, cap: usize) -> Self {
        let queue = Arc::new(OutputQueue::new(cap));
        queue.seed_initial(template);
        Self { queue }
    }
}

impl<T: Clone + Send + Sync + 'static> OutputStore for QueueStore<T> {
    fn output_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn borrow(&self, requested_ts: Instant) -> Option<(OutputPtr, BorrowGuard)> {
        let slot: Arc<Slot<T>> = self.queue.latest_at_or_before(requested_ts)?;
        let ptr: *const T = &slot.value as *const T;
        let guard = BorrowGuard::from_arc(slot);
        Some((OutputPtr(ptr as *const u8), guard))
    }

    unsafe fn commit(&self, ts: Instant, value_box_ptr: *mut u8) -> bool {
        // SAFETY: caller guarantees this was a Box<T> from alloc_fresh.
        let boxed: Box<T> = unsafe { Box::from_raw(value_box_ptr as *mut T) };
        self.queue.push(ts, *boxed)
    }

    fn alloc_fresh(&self) -> *mut u8 {
        // Clone the value from the latest committed slot (or the
        // seed-initial template, which is always present).  For stateful
        // outputs like `Series<T>` this propagates accumulated state.
        let latest = self
            .queue
            .latest_at_or_before(Instant::MAX)
            .expect("queue always has at least the seed_initial slot");
        let value: T = latest.value.clone();
        Box::into_raw(Box::new(value)) as *mut u8
    }

    unsafe fn drop_fresh(&self, ptr: *mut u8) {
        // SAFETY: caller guarantees this was a Box<T> from alloc_fresh.
        unsafe { drop(Box::from_raw(ptr as *mut T)) };
    }

    fn retire(&self) {
        self.queue.retire();
    }

    fn close(&self) {
        self.queue.close();
    }
}

// ===========================================================================
// Per-tick bookkeeping
// ===========================================================================

/// Readiness bookkeeping indexed by `tick_no mod W` for an operator node.
pub struct TickBookkeeping {
    /// Per-tick count of direct upstream edges that haven't signalled yet.
    /// Re-initialised to `direct_upstream_count` by the ingest loop at the
    /// start of each tick.
    pub remaining_inputs: Box<[AtomicUsize]>,
    /// Per-tick packed bitset of which input positions have fired.
    /// `Box<[AtomicU64]>` per tick-mod-W slot.
    pub incoming_bits: Box<[Box<[AtomicU64]>]>,
}

impl TickBookkeeping {
    pub fn new(pipeline_width: usize, arity: usize) -> Self {
        let words = arity.div_ceil(64).max(1);
        let incoming_bits: Vec<Box<[AtomicU64]>> = (0..pipeline_width)
            .map(|_| {
                (0..words)
                    .map(|_| AtomicU64::new(0))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect();
        let remaining_inputs: Vec<AtomicUsize> =
            (0..pipeline_width).map(|_| AtomicUsize::new(0)).collect();
        Self {
            remaining_inputs: remaining_inputs.into_boxed_slice(),
            incoming_bits: incoming_bits.into_boxed_slice(),
        }
    }

    /// Reset the tick slot to the initial count and clear bits.
    pub fn reset(&self, tick_slot: usize, count: usize) {
        self.remaining_inputs[tick_slot].store(count, Ordering::Relaxed);
        for word in self.incoming_bits[tick_slot].iter() {
            word.store(0, Ordering::Relaxed);
        }
    }

    /// Set a specific input-position bit for this tick slot.
    pub fn set_bit(&self, tick_slot: usize, input_pos: usize) {
        self.incoming_bits[tick_slot][input_pos / 64]
            .fetch_or(1u64 << (input_pos % 64), Ordering::Release);
    }

    /// Decrement remaining inputs by 1.  Returns the previous value.
    pub fn dec_remaining(&self, tick_slot: usize) -> usize {
        self.remaining_inputs[tick_slot].fetch_sub(1, Ordering::AcqRel)
    }

    /// Snapshot the bit words for compute.
    pub fn snapshot_bits(&self, tick_slot: usize) -> Vec<u64> {
        self.incoming_bits[tick_slot]
            .iter()
            .map(|a| a.load(Ordering::Acquire))
            .collect()
    }
}

// ===========================================================================
// Node
// ===========================================================================

/// A trigger edge: when this node's compute at tick `k` fires, it sets
/// bit `input_pos` on `downstream` at tick `k` and decrements the
/// downstream's remaining-inputs counter.
#[derive(Debug, Copy, Clone)]
pub struct TriggerEdge {
    pub downstream: usize,
    pub input_pos: usize,
}

/// A type-erased graph node.
pub struct Node {
    /// Graph position = topological rank.
    pub index: usize,
    /// Classification.
    pub kind: NodeKind,
    /// Output type id, for runtime checks.
    pub output_type_id: TypeId,

    // ------------------------------------------------------------------
    // Operator-only fields.  Unused for sources.
    // ------------------------------------------------------------------
    /// Compute function pointer (operators only).
    pub compute_fn: Option<ComputeFn>,
    /// Boxed operator state (operators only).
    pub state_ptr: StatePtr,
    /// Drop function for `state_ptr`.
    pub state_drop_fn: Option<DropFn>,
    /// Clone function for the output template (not used when we use the
    /// store's own alloc_fresh, but kept for completeness / debugging).
    pub clone_fn: Option<CloneFn>,
    /// Input node indices — each entry is `(upstream_node_idx, input_pos)`.
    pub input_edges: Box<[(usize, usize)]>,
    /// Per-edge type ids — for runtime validation at registration.
    pub input_type_ids: Box<[TypeId]>,
    /// Flat arity (total leaves).  Sources and consts have `arity == 0`.
    pub arity: usize,

    // ------------------------------------------------------------------
    // Source-only fields.  Ingest writes events into `source_rx_*` and
    // invokes `source_write_fn` with `output_scratch_ptr` as the target.
    // The scratch output is then handed to the store's commit path.
    // ------------------------------------------------------------------
    pub source_hist_rx_ptr: RawPtr,
    pub source_live_rx_ptr: RawPtr,
    pub source_poll_fn: Option<crate::source::PollFn>,
    pub source_write_fn: Option<crate::source::WriteFn>,
    pub source_rx_drop_fn: Option<DropFn>,
    pub source_output_scratch_ptr: StatePtr,
    pub source_output_drop_fn: Option<DropFn>,

    // ------------------------------------------------------------------
    // Fan-out and bookkeeping.
    // ------------------------------------------------------------------
    /// Downstream trigger edges.
    pub trigger_edges: Vec<TriggerEdge>,
    /// Output storage (auto-GC queue).
    pub output: Arc<dyn OutputStore>,

    /// Per-tick readiness.  Only populated for operator nodes with at
    /// least one upstream (`direct_upstream_count > 0`).
    pub bookkeeping: Option<TickBookkeeping>,
    /// Direct-upstream count at registration time (operators only).
    pub direct_upstream_count: usize,
    /// Count of direct upstreams that participate in the per-tick
    /// wavefront — i.e., non-const upstreams.  A const upstream (an
    /// operator with arity == 0) never signals, so its edges are
    /// excluded.  This is the value to which `remaining_inputs` is reset
    /// at the start of each tick.
    pub effective_upstream_count: usize,
    /// Does this node participate in the per-tick wavefront?  True for
    /// sources and for operators with arity > 0.  False for Const
    /// operators (arity == 0).
    pub participates_in_wavefront: bool,
    /// Latest committed tick for this node; used for the temporal
    /// self-edge readiness check.
    pub last_committed_tick: AtomicI64,
    /// Latest committed Instant — used by value() / telemetry.
    pub last_committed_ts: AtomicI64,

    /// Serialises concurrent compute tasks for the same node.  Same-tick
    /// tasks are already serialised by the tick counter; this mutex
    /// defends the `&mut State` invariant when the scheduler speculates.
    pub compute_mutex: Mutex<()>,
}

/// Raw pointer wrapper flagged as [`Send`] + [`Sync`] for storage in
/// nodes.  Operator state allocations are single-writer (a single tick at
/// a time, enforced by [`compute_mutex`](Node::compute_mutex)) and can be
/// safely shared as raw pointers.
#[derive(Copy, Clone)]
pub struct StatePtr(pub *mut u8);

unsafe impl Send for StatePtr {}
unsafe impl Sync for StatePtr {}

impl StatePtr {
    pub const NULL: Self = Self(std::ptr::null_mut());
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

/// [`Send`]+[`Sync`] wrapper for a raw pointer that is stable for the
/// lifetime of the graph (e.g. source-receiver heap allocations).
#[derive(Copy, Clone)]
pub struct RawPtr(pub *mut u8);

unsafe impl Send for RawPtr {}
unsafe impl Sync for RawPtr {}

impl RawPtr {
    pub const NULL: Self = Self(std::ptr::null_mut());
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}

impl Node {
    pub fn new_operator<T: Clone + Send + Sync + 'static>(
        index: usize,
        compute_fn: ComputeFn,
        clone_fn: CloneFn,
        state_ptr: *mut u8,
        state_drop_fn: DropFn,
        output_template: T,
        input_edges: Box<[(usize, usize)]>,
        input_type_ids: Box<[TypeId]>,
        arity: usize,
        queue_cap: usize,
        pipeline_width: usize,
    ) -> Self {
        let direct_upstream_count = input_edges.len();
        let bookkeeping = if direct_upstream_count > 0 {
            Some(TickBookkeeping::new(pipeline_width, arity))
        } else {
            None
        };
        let store: Arc<dyn OutputStore> = Arc::new(QueueStore::<T>::new(output_template, queue_cap));
        Self {
            index,
            kind: NodeKind::Operator,
            output_type_id: TypeId::of::<T>(),
            compute_fn: Some(compute_fn),
            state_ptr: StatePtr(state_ptr),
            state_drop_fn: Some(state_drop_fn),
            clone_fn: Some(clone_fn),
            input_edges,
            input_type_ids,
            arity,
            source_hist_rx_ptr: RawPtr::NULL,
            source_live_rx_ptr: RawPtr::NULL,
            source_poll_fn: None,
            source_write_fn: None,
            source_rx_drop_fn: None,
            source_output_scratch_ptr: StatePtr::NULL,
            source_output_drop_fn: None,
            trigger_edges: Vec::new(),
            output: store,
            bookkeeping,
            direct_upstream_count,
            // Filled in by graph.add_operator after it knows upstream kinds.
            effective_upstream_count: 0,
            participates_in_wavefront: arity > 0,
            last_committed_tick: AtomicI64::new(-1),
            last_committed_ts: AtomicI64::new(Instant::MIN.as_nanos()),
            compute_mutex: Mutex::new(()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_source<T: Clone + Send + Sync + 'static>(
        index: usize,
        hist_rx_ptr: *mut u8,
        live_rx_ptr: *mut u8,
        poll_fn: crate::source::PollFn,
        write_fn: crate::source::WriteFn,
        rx_drop_fn: DropFn,
        output_scratch_ptr: *mut u8,
        output_template: T,
        output_drop_fn: DropFn,
        queue_cap: usize,
    ) -> Self {
        let store: Arc<dyn OutputStore> = Arc::new(QueueStore::<T>::new(output_template, queue_cap));
        Self {
            index,
            kind: NodeKind::Source,
            output_type_id: TypeId::of::<T>(),
            compute_fn: None,
            state_ptr: StatePtr::NULL,
            state_drop_fn: None,
            clone_fn: None,
            input_edges: Box::new([]),
            input_type_ids: Box::new([]),
            arity: 0,
            source_hist_rx_ptr: RawPtr(hist_rx_ptr),
            source_live_rx_ptr: RawPtr(live_rx_ptr),
            source_poll_fn: Some(poll_fn),
            source_write_fn: Some(write_fn),
            source_rx_drop_fn: Some(rx_drop_fn),
            source_output_scratch_ptr: StatePtr(output_scratch_ptr),
            source_output_drop_fn: Some(output_drop_fn),
            trigger_edges: Vec::new(),
            output: store,
            bookkeeping: None,
            direct_upstream_count: 0,
            effective_upstream_count: 0,
            participates_in_wavefront: true, // sources always participate
            last_committed_tick: AtomicI64::new(-1),
            last_committed_ts: AtomicI64::new(Instant::MIN.as_nanos()),
            compute_mutex: Mutex::new(()),
        }
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // Drop operator state.
        if !self.state_ptr.is_null()
            && let Some(drop_fn) = self.state_drop_fn
        {
            unsafe { drop_fn(self.state_ptr.0) };
            self.state_ptr = StatePtr::NULL;
        }
        // Drop source receivers + output scratch.
        if !self.source_hist_rx_ptr.is_null()
            && let Some(drop_fn) = self.source_rx_drop_fn
        {
            unsafe { drop_fn(self.source_hist_rx_ptr.0) };
            self.source_hist_rx_ptr = RawPtr::NULL;
        }
        if !self.source_live_rx_ptr.is_null()
            && let Some(drop_fn) = self.source_rx_drop_fn
        {
            unsafe { drop_fn(self.source_live_rx_ptr.0) };
            self.source_live_rx_ptr = RawPtr::NULL;
        }
        if !self.source_output_scratch_ptr.is_null()
            && let Some(drop_fn) = self.source_output_drop_fn
        {
            unsafe { drop_fn(self.source_output_scratch_ptr.0) };
            self.source_output_scratch_ptr = StatePtr::NULL;
        }
    }
}

// SAFETY: Node owns only type-erased heap allocations protected by the
// invariants documented on each field.  All concurrent access is mediated
// by atomic fields, `compute_mutex`, or single-writer discipline
// (source_*_ptr is read only by the ingest task).
unsafe impl Send for Node {}
unsafe impl Sync for Node {}
