# Wavefront Execution Model — Design

## Motivation

The current `Scenario` runtime processes the computation graph **sequentially**: one
timestamp at a time, one node at a time within each flush (`src/scenario/graph.rs`).
The 2D grid of `(node, tick)` has parallelism on two independent axes — nodes
at the same timestamp with no data dependencies, and independent timestamps for
stateless operators — but neither is exploited.

Operators fall into two categories:

- **Stateless**: output at time _t_ depends only on inputs at time _t_. The
  `State` is either `()` or an immutable configuration captured at init time
  (closures, precomputed index maps). Analogous to attention layers in
  transformers.
- **Stateful**: output at time _t_ depends on state carried forward from time
  _t−1_. The `State` mutates across ticks (ring buffers, running sums, EMA
  accumulators). Analogous to RNN cells.

The wavefront model exploits this distinction: stateless operators have no
vertical (temporal) dependency, so all their timestamps are independent once
their inputs are ready.

## Architecture

### Scheduling unit

Each operator node is its own scheduling unit. A cell `(node, tick)` is ready
when:

- **Horizontal**: all of the node's upstream inputs have completed at the same tick
- **Vertical**: if the operator is stateful, the same node has completed at tick−1

Both are tracked uniformly as a dependency counter per cell. When any node
completes at tick `t`, it decrements the counter for each downstream node at
tick `t` (horizontal notification) and for itself at tick `t+1` if stateful
(vertical notification). A counter reaching zero enqueues the cell into a
ready queue. No grid scanning is needed.

### State and Output as a unified pair

An operator's conceptual "state" includes both its `State` (internal) and its
`Output` (visible to downstream). For a stateful operator like `ForwardFill`
(`State = ()`), the state lives entirely in the output. The runtime therefore
manages both as a pair:

- **Stateless**: each tick clones fresh from the init `(state, output)` pair.
  After compute, the state is dropped and the output is pushed to the
  versioned ring.
- **Stateful**: tick 0 clones from init; tick _t+1_ clones from the `(state,
  output)` values produced by tick _t_. After compute, the ring receives a
  clone of the output, and the current pointers are updated for the next tick.

### Versioned storage

Every operator node stores its per-tick output in a `VersionedRing`
(`VecDeque`-backed). Entries are pushed as they are computed. During the
wavefront, GC is deferred — entries are preserved for final inspection via
`value_at()`. A monotonic `low_water` cursor tracks the minimum in-flight tick
across all nodes, providing the basis for future auto-pop of stale entries.

### Execution loop

```
Setup:
  - Pre-fill source node rings for all ticks.
  - Set dep[c][t] = input_count + (1 if stateful and t > 0).
  - Source nodes notify downstream → decrement → enqueue at 0.

Loop:
  while let Some((tick, node)) = ready_queue.pop_front():
    execute_node(node, tick)
    notify downstream (same tick)
    notify self (next tick, if stateful)
```

`execute_node` resolves input pointers from upstream versioned rings for the
current tick, clones the `(state, output)` pair, calls the erased compute
function, pushes the output to the ring, and updates the current pointers for
stateful nodes.

## Public API

```rust
let mut sc = WavefrontScenario::new();

let h_a = sc.add_source(ArraySource::new(timestamps, values, shape, default));
let h_b = sc.add_source(ArraySource::new(timestamps, values, shape, default));

let h_add = sc.add_operator(Add::new(), (h_a, h_b));
let h_diff = sc.add_operator(Diff::new(), h_add);

sc.run();

assert_eq!(sc.value_at::<Array<f64>>(h_diff, 2).as_slice(), &[...]);
```

## Traits

### `experimental::Operator`

```rust
pub trait Operator: 'static {
    type State: Send + Clone + 'static;
    type Inputs: InputTypes + Sized;
    type Output: Send + Clone + 'static;

    fn init(self, inputs: ..., timestamp: Instant) -> (Self::State, Self::Output);
    fn compute(state: &mut Self::State, inputs: ..., output: &mut Self::Output,
               timestamp: Instant, produced: ...) -> bool;

    fn is_stateful() -> bool { false }
}
```

Differences from `crate::Operator`:
- `State: Clone` and `Output: Clone` — needed for cloning per parallel tick
- `Inputs: Sized` — PoC simplification (no variadic `[Input<T>]` inputs)
- `is_stateful()` — drives vertical dependency edges

### `experimental::Source`

```rust
pub trait Source: 'static {
    type Output: Send + Clone + 'static;
    type Value: Send + 'static;

    fn events(&self) -> Vec<(Instant, Self::Value)>;
    fn init_output(&self) -> Self::Output;
    fn write(value: &Self::Value, output: &mut Self::Output);
}
```

Batch source — all events known upfront. No async channels.

### `WavefrontAdapter<O>`

Bridges any `crate::Operator` impl to `experimental::Operator` (requires
`State: Clone`, `Output: Clone`, `Inputs: Sized`). Conservatively treats all
adapted operators as stateful.

## Module structure

```
src/experimental/
  mod.rs          — module docs, re-exports
  operator.rs     — Operator trait
  source.rs       — Source trait + ArraySource
  storage.rs      — VersionedRing (VecDeque)
  graph.rs        — WavefrontNode + WavefrontGraph
  scheduler.rs    — ready queue, dep counters, low-water tracking
  scenario.rs     — WavefrontScenario public API + tests
  adapter.rs      — WavefrontAdapter<O: crate::Operator>
  operators/
    add.rs        — Add (stateless, State = ())
    multiply.rs   — Multiply (stateless, State = ())
    diff.rs       — Diff (stateful, State = DiffState)
    id.rs         — Id (stateless, State = ())
```

## Scope and limitations

**In this PoC:**
- Per-node scheduling with uniform dependency tracking
- Versioned output rings
- Historical replay only (batch source, all events known upfront)
- Fixed-arity operators (`Inputs: Sized`)
- Four demonstration operators

**Deferred:**
- Rayon thread-pool parallelism across independent cells
- Variadic-input operators (`[Input<T>]`)
- `Series<T>` outputs and Record operator
- Clock-gated operators (`Clocked`)
- Python operator support in wavefront
- Live mode / async source channels
- Ring buffer auto-pop with proper downstream consumer tracking
