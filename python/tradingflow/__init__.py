"""Python operators for the TradingFlow engine.

Each operator module defines a segment mirroring the Rust `Segment` trait
method-for-method, and binds it to `__op__` (or defines `build(**kwargs)` when
the segment takes construction parameters):

    init(self, inputs) -> state
    reset(self, inputs, state) -> outputs
    compute(self, inputs, state, instant) -> outputs

`inputs` is a single tuple with one owned NumPy array per wired leaf, in tree
order; signals arrive as rank-0 bool arrays, so a pulse answers `if signal:`
directly. `instant` is naive nanoseconds. `outputs` mirrors the output
interface — a bare array for a single port, a sequence for a tuple of them.

`reset` runs once at build time, where its first return sizes the node's output
buffers, and again after every generation. It must return the segment's
quiescent outputs: signals cleared, values either retained or defaulted.

Arrays cross the boundary **by copy** in both directions, so everything handed
to a segment is an ordinary owned NumPy array — free to keep in `state` across
generations, mutate in place, or hand to another thread. Nothing here borrows
engine memory, and nothing returned to the engine stays aliased.

An operator that needs history keeps it itself: only the latest cross-section
crosses the boundary, never a series.

| Package | Contents |
| --- | --- |
| `metric` | evaluation of predictions against realized targets |
| `predictor` | cross-sectional mean and covariance models |
| `portfolio` | position sizing from predicted moments |

`numpy` is always importable; `scipy` and `cvxpy` are needed by some operators
and imported where they are used.
"""
