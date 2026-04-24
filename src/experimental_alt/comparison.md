# Wavefront PoC comparison — `experimental/` vs `experimental_alt/`

Two independent proofs-of-concept for the wavefront execution model live
in this repository:

- `src/experimental/` — DeepSeek Pro V4's attempt.
- `src/experimental_alt/` — this module (co-authored with the user).

This document compares the two and calls out correctness concerns in
both.

## 1. High-level framing

| | `experimental/` | `experimental_alt/` |
|---|---|---|
| **Execution model** | Offline / batch. All timestamps known upfront; full 2D `(node, tick)` grid materialised. | Streaming. Events arrive on async channels, tick count emerges dynamically. |
| **Scheduler** | Single-threaded `VecDeque` drain; parallelism deferred. | Real crossbeam-deque work-stealing pool + `std::thread` workers. |
| **Stateful/stateless** | Explicit `is_stateful()` method → stateful gets a vertical self-edge, stateless does not. | Conservative: every node gets an implicit self-edge. |
| **Dep tracking** | `dep_remaining[node][tick]` — O(V × T) 2D array pre-materialised at run start. | Per-node `[AtomicUsize; W]` indexed by `tick mod W`; `W` = pipeline depth (default 8). |
| **Storage** | `VersionedRing` — `VecDeque<(tick, *mut u8, drop_fn)>`. Manual low-water GC (deferred). | `OutputQueue<T>` — `Mutex<VecDeque<Arc<Slot<T>>>>`. Auto-GC via `Arc` refcount under capacity pressure. |
| **Source contract** | New synchronous batch trait — `events() -> Vec<(ts, value)>`. | Reuses the existing tokio mpsc `Source` unchanged. |
| **Input tree** | `Inputs: Sized` — excludes `[Input<T>]` slices. | `Inputs: InputTypes + ?Sized` — full variadic support. |
| **Trait bounds** | `State: Clone + Send`, `Output: Clone + Send`. | `State: Send`, `Output: Clone + Send`. |
| **Operators ported** | 4 (Add, Multiply, Diff, Id). | 9 (Const, Id, Add, Filter, Record, Lag, Rolling, Clocked, ConcatSync). |
| **Tests** | 5 integration. | 7 queue + 8 scenario. |

## 2. Pros — `experimental/` (DeepSeek)

1. **Cleaner conceptual model.** The stateful/stateless distinction is
   explicit in the type signature (`is_stateful()`) with direct
   operational meaning: stateless nodes have no vertical edge, so their
   ticks can in principle run in any order or in parallel.
   `experimental_alt/` throws this away via the conservative self-edge.
2. **Simpler storage primitive.** `VersionedRing` is straight raw
   pointers with drop-fns — no `Mutex`, no `Condvar`, no `Arc`. Much
   easier to reason about.
3. **Historical inspection for free.** `value_at(h, tick)` works
   trivially because all snapshots are kept until a later GC pass.
   `experimental_alt/` retires aggressively under capacity pressure.
4. **Single-threaded determinism.** Zero threads, zero atomics, zero
   races. Bugs are reproducible; stepping through in a debugger works.
5. **Clear path to parallelism.** The 2D grid with explicit dep counters
   is a textbook starting point for later swapping in a Rayon-style work
   queue. The hard reasoning work is done.
6. **`WavefrontAdapter<O>` migration path.** Existing `crate::Operator`
   impls plug in without being rewritten (when they satisfy
   `Clone + Sized`).

## 3. Cons — `experimental/` (DeepSeek)

1. **Not actually parallel.** The PoC is single-threaded. The design
   admits parallelism but doesn't demonstrate it.
2. **Multi-source timestamp bug (fatal).** In `scenario.rs:42-46`,
   merging timestamps only pushes when `ts > self.timestamps.last()`.
   If source A has events at `[1, 2, 3]` and source B has events at
   `[1, 2, 4]`, after adding A `timestamps = [1, 2, 3]`; adding B
   appends only `4`, giving `[1, 2, 3, 4]`. But B's versioned ring was
   pre-filled with only 3 entries (B's own events) at tick indices
   0/1/2, while the scheduler later looks up tick 3 in B's ring →
   `get(3)` returns `None` → panic in `scheduler.rs:106`. This rules
   out any graph where sources don't share exactly the same timestamp
   set. All current tests happen to share timestamps, masking it.
3. **Tick-index-as-lookup-key is brittle.** `VersionedRing::get(tick: usize)`
   uses the scheduler's integer tick. For sources to be looked up
   correctly, their pre-fill must align one-to-one with the global tick
   space — which the merge bug above shows is fragile.
4. **Batch-only sources.** No live streaming; must know all events
   upfront. Breaks the existing async use case.
5. **`Inputs: Sized` kills variadic ops.** Concat / Stack / ConcatSync —
   exactly the operators that most benefit from wavefront scheduling
   in a per-symbol portfolio graph — cannot be expressed.
6. **No Series / Record / Clocked / Filter.** The operator surface is
   very thin. The tests exercise three archetypes (stateless binary,
   stateful self-loop, and the combination).
7. **O(V × T) dep array.** Fine for T in the hundreds, problematic for
   streams of millions of events. Nothing in the design says "stream";
   it's fundamentally a batch schedule.
8. **`State: Clone` bound is heavy.** Any operator whose state contains
   a non-Clone closure or complex structure can't implement the trait.
   `Filter` (captures a predicate closure) would need the closure to be
   `Clone`.
9. **Manual raw-pointer vtable per node.** `state_clone_fn` /
   `state_drop_fn` / `output_clone_fn` / `output_drop_fn` / `compute_fn`
   — five function pointers stored per node, plus `current_*` and
   `init_*` pointer fields kept consistent by imperative mutation
   during `execute_node`. More surface for lifecycle bugs.
10. **`current_*_ptr` management on gate failures is unclear.**
    `execute_node` unconditionally clones and writes to the ring, then
    updates `current` for stateful ops. What happens if a future
    `Filter`-style op returns `false`? No handling path.

## 4. Pros — `experimental_alt/` (this module)

1. **Actually parallel.** crossbeam-deque + `std::thread` workers run
   concurrently; ingest drives on tokio in parallel with compute.
   Visible behaviour, not just design.
2. **Preserves the existing async `Source`.** No new trait, no
   batch-only restriction; live streams are supported.
3. **Full `InputTypes` expressivity.** Variadic `[Input<T>]` tuples,
   12-arity heterogeneous tuples, everything the legacy runtime
   supports.
4. **Bounded memory via W-indexed ring + cap.** Independent of T;
   suitable for long streams.
5. **Auto-GC via `Arc` refcount is correct without manual
   bookkeeping.** No low-water tracking needed.
6. **Broader operator coverage.** Record / Rolling / Lag (Series
   flow), Filter (return-bool gating), Clocked (produced-bit gating),
   ConcatSync (slice inputs + selective per-position copy) — every
   scheduler code path exercised.
7. **Less-heavy trait bounds.** Only `Output: Clone` required; `State`
   is just `Send`.

## 5. Cons — `experimental_alt/` (this module)

1. **Conservative self-edge for all.** Every node — even pure `Add` —
   serialises its own time axis. A hypothetical future "stateless
   operator can run ticks concurrently" optimisation is blocked by
   this choice. DeepSeek's `is_stateful()` gets this right in
   principle.
2. **Every-node-at-every-tick ingest dispatch.** For each tick, ingest
   signals every operator with bookkeeping via firing-or-not-firing
   trigger edges. O(|source edges|) per tick.  Sparse-activity graphs
   pay for nothing.
3. **`wait_for_commit` is a polling loop.**
   `tokio::time::sleep(1ms)` in a retry loop — correct but crude.
   Should be a condvar keyed on min-committed-tick.
4. **Significantly more moving parts.** `Mutex` + `Condvar` + `Arc` +
   two atomics per slot + type-erased trait object for storage +
   worker pool + ingest future + task deque. Harder to audit than a
   single-threaded `VecDeque` loop.
5. **Series outputs are per-tick deep clones.** O(t²) worst-case
   memory before GC for a `Record` operator — DeepSeek's approach has
   the same property via snapshots. A real v2 needs atomic-commit
   shared-buffer Series.
6. **`latest_at_or_before` is an O(n) linear scan.** Fine when `n` is
   bounded by `pipeline_width`; would need a `BTreeMap` for large
   queues.
7. **No shutdown propagation on worker panic.** `std::thread::spawn`'s
   panic is swallowed until join; a compute panic can silently
   deadlock the drain.
8. **Subtle "upstream advances before downstream borrows" race** —
   prevented in the current design by capacity-pressure eviction only
   popping the *oldest* slot (never the newest), and by
   `pipeline_width ≥ 2`. It took some thinking to rule out; correct
   under the invariants but more fragile to reason about than
   DeepSeek's "keep everything until GC" model.

## 6. Where each is the right answer

- **`experimental/` (DeepSeek) wins if**: you are doing batch quant
  research with all data loaded upfront; you want an eventual parallel
  scheduler but correctness-first today; you want historical snapshots
  for visualisation/debugging; your graphs are small and use
  fixed-arity operators only.
- **`experimental_alt/` (mine) wins if**: you need live streaming
  alongside batch; you need variadic/slice-input operators like
  ConcatSync for per-symbol pipelines; you need actual thread-level
  parallelism now; you want bounded memory regardless of run length.

## 7. Synthesis — the best end state probably takes from both

1. **From `experimental_alt/`**: async `Source` + bounded-memory
   `OutputQueue` + crossbeam worker pool — this is the hard part to
   get right and is now working.
2. **From `experimental/`**: explicit `is_stateful()` — drop the
   conservative self-edge for stateless ops, unlocking the second axis
   of parallelism without compromising correctness.
3. **From `experimental_alt/`**: auto-`Arc` GC over manual low-water
   tracking — fewer knobs, provably correct under the invariants.
4. **From `experimental/`**: `WavefrontAdapter` — a no-cost bridge from
   the existing trait makes the remaining-operators port trivial, at
   the price of the `Clone` bounds.

The multi-source timestamp merge bug in `experimental/scenario.rs:42-46`
is the single most important issue across both implementations and
needs to be fixed before DeepSeek's 2D-batch model can serve real
graphs.
