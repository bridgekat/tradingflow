# Wavefront runtime — design notes

This document summarises the conversation that produced this module and
the design choices embedded in the code at `src/experimental_alt/`.  It
is a proof-of-concept; the companion plan lives at
`~/.claude/plans/now-plan-a-rewrite-iridescent-steele.md`.

## 1. Motivation

The legacy runtime (`src/scenario/`) drives a computation graph strictly
serially: `Graph::flush` pops a min-heap of ready nodes in topological
order and runs each operator's `compute` on the calling thread.  For
quant-research DAGs with wide per-symbol fan-out and deep stateful
chains this leaves two separate axes of parallelism unused:

1. **Node-axis** — within a single timestamp `t`, independent DAG
   branches could run concurrently.  Each node already owns its
   `&mut State` and output buffer, so no operator-level locks are
   needed; only scheduler bookkeeping is contended.
2. **Time-axis** — `t` and `t+1` could overlap if a writer at `t+1`
   didn't clobber a reader of `t` in flight.  Today's single-buffer
   output convention is the only real blocker — stateful vs stateless
   operators behave the same with respect to buffer aliasing.

## 2. Key framing: what the stateful/stateless distinction actually means

The initial question was whether stateful operators (analogous to RNNs)
vs stateless ones (analogous to attention) need to be scheduled
differently.  The conclusion from the conversation:

- **Within one flush**, stateful vs stateless makes no difference.  Each
  node has a single `&mut State` that is disjoint from every other
  node's.  A wavefront of independent nodes parallelises cleanly for
  both.
- **Across `t`**, only operators with state carry a temporal self-edge
  (`(t-1, i) → (t, i)`).  A purely stateless op could run `(t, i)` and
  `(t+1, i)` concurrently — except the single output buffer prevents it
  regardless of state.
- **The dominant axis is therefore node-axis**, and the dominant trick
  for unlocking cross-`t` pipelining is not state handling but
  **versioned output storage**.  The auto-managed queue design falls
  out of this.

## 3. Reframing as a 2D DAG

The runtime sees a dependency DAG over tasks `(t, node_id)`:

- **Spatial edges**: `(t, i) → (t, j)` whenever `i → j` in the graph.
- **Temporal self-edges**: `(t-1, i) → (t, i)` — required for stateful
  nodes, and adopted **unconditionally** in this PoC to keep the scheduler
  simple.  It costs a small amount of cross-`t` parallelism for
  stateless nodes; we judged this acceptable for a first pass.
- **Buffer-reuse edges** (implicit): a reader of slot `(t, i)` must
  release before a writer at `(t + K, i)` may reuse slot `t mod K`.
  The output queue handles this through `Arc` refcounts.

Sources ingest events in timestamp order; each coalesced tick becomes a
wavefront that propagates through the graph.  The ingest loop runs on a
tokio runtime; compute runs on a `std::thread` worker pool.

## 4. Subgraph units — deferred

A major design thread in the conversation was user-declared subgraph
units as the scheduling granularity (to amortise per-op overhead for
cheap operators and give users explicit control).  On agreement this
was **dropped from the PoC**: single-node scheduling was chosen to
minimise initial surface area.  The traits were kept composition-ready
so a later `Scenario::subgraph(|s| { ... })` builder can lower to an
`ErasedOperator` whose `State` is a nested mini-graph — no trait-shape
changes required.

## 5. Scope decisions (all answered during planning)

- **Runtime scope**: full wavefront — node-axis parallel **and**
  cross-`t` pipelining.  Not just a trait rewrite.
- **Subgraph units**: deferred (see §4).
- **Python bridge**: Rust-only PoC.  The existing `NativeScenario`
  keeps using `src/scenario`.
- **Relation to existing traits**: standalone.  No
  `impl<O: old::Operator> new::Operator for O` adapter.  A targeted
  subset of operators was ported directly under the new traits.
- **Scheduling unit**: one node per dispatch.
- **Output buffering**: auto-managed queue — writers push, oldest slots
  evicted under capacity pressure when no reader still holds them.

## 6. Trait design

### 6.1 `Operator`

Signature-identical to `crate::operator::Operator` except
`type Output: Clone`.  The scheduler clones the latest committed slot
per timestamp to give `compute` a fresh buffer to write into; for
replacement-style outputs (`Array<T>`) the clone content is overwritten
anyway, while for accumulator-style outputs (`Series<T>`) the clone
propagates prior state so operators like `Record` see accumulated
history.

```rust
pub trait Operator: Send + 'static {
    type State:  Send + 'static;
    type Inputs: InputTypes + ?Sized;
    type Output: Clone + Send + 'static;
    fn init(self, inputs: Refs<'_>, ts: Instant) -> (Self::State, Self::Output);
    fn compute(
        state:    &mut Self::State,
        inputs:   Refs<'_>,
        output:   &mut Self::Output,
        ts:       Instant,
        produced: Produced<'_>,
    ) -> bool;
}
```

Existing operator bodies port verbatim.  `Const` additionally gains the
`T: Clone + Sync` bound.

### 6.2 `Source`

Re-exported unchanged from `crate::source`.  The tokio `mpsc`-channel
contract, `PeekableReceiver`, and the `PollFn`/`WriteFn` type-erased
pointers all stay.  The `drain_hist` early-exit + live-clamping +
coalescing logic in `src/scenario/queue.rs` is subtle and rewriting it
would be pure cost — so the ingest adapter reuses the same merge loop
almost verbatim.

### 6.3 `InputTypes` machinery

Re-exported unchanged.  The nested `Refs<'_>` / `Produced<'_>` trees
(`Input<T>`, tuples up to arity 12, trailing `[T]` slice) are
load-bearing for `Clocked`, `ConcatSync`, `ForwardAdjust`, `Volatility`
and are not what we're changing.

## 7. Output storage

`OutputStore` is a small type-erased trait with one concrete
implementation — `QueueStore<T>` backed by `OutputQueue<T>`.  The
original plan foresaw separate `Series`-specific storage; the PoC
collapsed this into a single queue type and accepts the memory cost of
per-tick snapshots (see §11.2).

### 7.1 `OutputQueue<T>`

`Mutex<VecDeque<Arc<Slot<T>>>>` + `Condvar`.

- **Write**: lock; if `len < cap`, push and notify.  If at cap, evict
  the oldest slot iff its `Arc::strong_count == 1` (no live reader);
  else park on the condvar.
- **Read**: lock; scan from the back for the first slot with `ts <=
  requested_ts`; clone its `Arc`.  The reader holds the clone for the
  duration of compute, preventing eviction.
- **No proactive retire.**  The initial design had `retire()` pop front
  slots whenever their refcount returned to 1.  This was wrong: a
  downstream reader at tick `k` might request `slot_k` only after its
  upstream has already advanced to tick `k+1` and retired `slot_k`.
  Lazy eviction (under capacity pressure only) was the fix.

### 7.2 `alloc_fresh` clones the latest committed slot

Not the seed template.  For `Record`/`Lag`/`Rolling` this is what
propagates state: each tick's output buffer begins life as a deep clone
of the previous tick's output, compute mutates it, and the result is
committed as a new slot.  Auto-eviction reclaims older snapshots once
no reader holds them.

## 8. Per-node bookkeeping

Indexed by `tick_no mod W`, where `W` = scenario-level
`pipeline_width` (default 8):

- `remaining_inputs: Box<[AtomicUsize; W]>` — count of direct upstream
  edges that haven't yet signalled for this tick.  Re-initialised by
  the ingest loop at the start of each tick to `effective_upstream_count`
  (excludes Const upstream edges, which never signal).
- `incoming_bits: Box<[AtomicU64 words; W]>` — packed bitset of which
  input positions have fired this tick.  Cleared by the operator task
  at end of compute.
- `last_committed_tick: AtomicI64` — drives the temporal self-edge
  readiness check.

`(t mod W)` indexing + `pipeline_width` throttling in ingest prevents
in-flight ticks from colliding on the same slot.

## 9. Scheduler

`crossbeam_deque::Injector` + per-worker `Worker`/`Stealer` on plain
`std::thread`.  N workers = `std::thread::available_parallelism()`.
`Task = (node_idx, tick_no, ts)`.

**Readiness predicate** for `(t, node)`:

```
(1) all upstream edges have signalled at t
    — enforced by remaining_inputs[t mod W].fetch_sub(1) == 1
      (the decrementer that sees 1 enqueues the task)
(2) node.last_committed_tick >= t - 1
    (temporal self-edge; checked at worker pickup)
(3) output queue can accept a new slot
    (enforced inside push())
```

Only (1) triggers enqueue.  (2) is checked inside the worker; if
unsatisfied, the task is re-injected to the global deque and the worker
moves on.  (3) is handled by the `push()` blocking path, which parks on
the queue's condvar.  Same-node tasks serialise naturally through (2)
without needing a per-node mutex — the `fetch_sub → 1` guarantees
exactly one task per `(node, tick)` and the temporal self-edge enforces
in-order execution.

`rayon` was explicitly rejected: its scope-based fork-join model does
not admit tasks from multiple in-flight timestamps coexisting on a
queue, which is the whole point of cross-`t` pipelining.

## 10. Ingest

Adapted — not verbatim — from `src/scenario/queue.rs`.  The hist/live
merge, `drain_hist` early-exit, live-timestamp clamping, and
coalescing all carry over unchanged.  The transformation:

- Instead of a synchronous `self.graph.flush(qts, &firing_sources)`,
  dispatch:
  1. Throttle: wait until every wavefront-participating node has
     committed tick `tick_no - W`, freeing the slot for reuse.
  2. Reset every wavefront-participant operator's `remaining_inputs`
     and `incoming_bits` at `tick_no mod W` to the node's
     `effective_upstream_count`.
  3. For each **firing** source: write the event into a fresh output
     buffer via its `write_fn`, commit to its queue, then fire its
     trigger edges into downstreams (set bit, decrement counter,
     enqueue task if counter hits 0).
  4. For each **non-firing** source: fire its trigger edges
     **without** setting bits (counter decrement only).  Downstream
     still runs and sees no-fire on that input.
- Ingest proceeds to the next tick without waiting for compute to
  drain, enabling cross-`t` pipelining up to width `W`.

## 11. Known limitations and trade-offs

### 11.1 Conservative self-edge for all nodes

Every node — including stateless ones — serialises its own time axis.
A hypothetical `Output: Clone + Stateless` optimisation that lets
`t` and `t+1` run concurrently for a pure element-wise `Add` was
judged out of scope.  The win from this would be marginal on top of
node-axis parallelism for real graphs.

### 11.2 Series outputs snapshot per tick

`Record` produces `Series<T>` whose cloned-per-tick semantics
inflate memory to O(t²) in the worst case before eviction kicks in.
For the tests here (ticks in the hundreds) this is tolerable; for
larger scenarios the target v2 is a shared append-only backing buffer
with an atomic commit cursor, as originally outlined in the plan.

### 11.3 Every node at every tick

The ingest loop sends a signal to every wavefront-participating
operator at every tick — either via a firing source's trigger edge or
via a non-firing source's no-fire decrement.  This is O(|edges_from_sources|)
per tick.  Reachability-based scheduling was considered and deferred;
for the tests here the uniform approach is simpler and correct.

### 11.4 Deferred scope

- PyO3 / `NativeScenario` exposure.
- `CsvSource`, `IterSource`, stocks/metrics operator families,
  `Resample`, `Where`, `Apply`/`Map` variants, non-sync Concat/Stack,
  `Cast`, `Select`.
- Subgraph builder.
- Timing-based `tests/pipelining.rs` and `tests/fanout.rs` from the
  plan — the functional fan-out test is present
  (`wide_fanout_all_branches_produce`), but wall-clock speedup
  assertions are brittle and were skipped for the first pass.
- Loom-based scheduler fuzzing.

## 12. Module layout

```
src/experimental_alt/
├── mod.rs                   — crate-level docs + re-exports
├── data.rs                  — re-exports of crate::data (Array, Series, InputTypes, …)
├── source.rs                — re-exports of crate::source (Source, ErasedSource)
├── operator.rs              — new Operator trait + ErasedOperator
├── queue.rs                 — OutputQueue<T> with Arc-refcount auto-GC
├── scenario/
│   ├── mod.rs               — Scenario facade (add_source/operator, value, run)
│   ├── handle.rs            — Handle<T> + InputTypesHandles trait tree
│   ├── node.rs              — Node, OutputStore trait, QueueStore, bookkeeping
│   ├── graph.rs             — Graph: nodes + trigger-edge index + seal + value
│   ├── scheduler.rs         — crossbeam-deque workers + readiness predicate
│   └── ingest.rs            — adapted source-merge loop + per-tick dispatch
├── operators/
│   ├── mod.rs
│   ├── const.rs             — zero-input permanent-slot operator
│   ├── id.rs                — single-input clone passthrough
│   ├── add.rs               — element-wise Array<T> addition
│   ├── filter.rs            — predicate-gated return-bool propagation
│   ├── record.rs            — Array<T> → Series<T> append
│   ├── lag.rs               — stateful N-step-back lookup
│   ├── rolling.rs           — count-window rolling mean
│   ├── clocked.rs           — clock-gated wrapper around an inner op
│   └── concat_sync.rs       — variadic slice input + per-position produced bits
└── sources/
    └── mod.rs               — re-exports ArraySource, clock
```

`Cargo.toml` gained `crossbeam-deque = "0.8"` and `crossbeam-utils =
"0.8"`.  `src/lib.rs` gained `pub mod experimental_alt;`.  Nothing else
changed outside this module.

## 13. Tests

- **7 queue unit tests**: seed/read, push+read-latest, lazy retire,
  capacity-pressure eviction, live-reader pinning, writer park-on-full,
  close-wakes-blocked-writer.
- **8 scenario integration tests** (all `tokio::test`): single-source
  Record, source + Const + Add + Record, two-source coalesced Add +
  Record, Rolling mean over a Recorded Series, Lag, Filter gating,
  Clocked-Filter clock gating, wide-fan-out of 8 independent Add+Record
  branches.

Together these exercise every scheduler code path: stateless chains,
stateful operators with the temporal self-edge, `Series<T>` flow,
return-bool gating, produced-bit gating, node-axis parallelism, and
multi-source coalescing.

Full `cargo test --lib` passes 249 tests — no regression in the existing
runtime.

## 14. Intended next steps (if this design proves out)

1. Shared append-only `Series` storage (atomic commit cursor) to
   eliminate the per-tick clone.
2. Subgraph builder — user-declared scheduling units.
3. Port the rest of the operator library.
4. Loom fuzzing of the scheduler / queue.
5. PyO3 exposure behind the existing `python` feature.
6. Replace `src/scenario/` once the experimental runtime has parity.
