"""Evaluation metrics, as Python segments hosted by the Rust engine.

Each module defines a segment whose contract mirrors the Rust `Segment`
trait, and binds it to `__op__` (or defines `build(**kwargs)` when the
segment takes construction parameters):

    init(self, inputs) -> state
    reset(self, inputs, state) -> outputs
    compute(self, inputs, state, instant) -> outputs

`inputs` is a tuple with one owned NumPy array per wired leaf, in tree
order; signals are rank-0 bool arrays. `outputs` mirrors the output
interface — a bare array for a single port, a sequence for a tuple of
them. `instant` is the event time in naive nanoseconds.

`reset` is called once at build time and again after every generation, and
must return the node's quiescent outputs: signals cleared, values either
retained or defaulted, per the segment's own semantics.

Arrays cross the boundary by copy in both directions, so they are ordinary
owned NumPy arrays — free to keep in `state` across generations, mutate, or
hand to other threads.
"""
