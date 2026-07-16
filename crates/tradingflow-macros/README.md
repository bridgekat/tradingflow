# `tradingflow-macros`

## The `segment!` macro

A `segment!` expression creates a new `Segment` by wiring together smaller `Segment`s. It reads like a closure:

```rust,ignore
let seg = segment!(|a: Port<i64>, b: Port<i64>| -> (Port<i64>, Port<i64>) {
    let c = add() @ (a, b);  // c = a + b
    let d = inc(1) @ c;      // d = c + 1
    (d, c)                   // two outputs
});
```

The header names each input wire with its interface type, and then to the right of `->` the output wire interface type. The body is a sequence of `let` bindings, ending in a result wire expression.

Segments apply to wires prefix-style with `@` — the expression left of `@` is the segment, the wires to its right are its inputs.

Applications nest inside any wire expression and chain right-associatively, each nesting desugaring to a fresh intermediate wire:

```rust,ignore
segment!(|x: Port<i64>| -> Port<i64> {
    add() @ (x, inc(1) @ x)  // let t = inc(1) @ x; add() @ (x, t)
});
```

## Module path override

Expansions prefix the combinators by `::tradingflow_graph::typed`, so callers need `tradingflow-graph` among their dependencies. A leading `@[path]` overrides that path:

```rust,ignore
segment!(@[::some::path::to::module] |a: Port<i64>| -> Port<i64> { /* ... */ });
```

will resolve combinators in `::some::path::to::module` instead of `::tradingflow_graph::typed`.
