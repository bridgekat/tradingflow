"""Clocked operator transformer — gates a Python operator on a clock input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..operator import Operator
from ..types import Handle


@dataclass
class ClockedState:
    """State for [`Clocked`]."""

    inner_state: Any
    # Stored as an unbound function to avoid per-call attribute lookup.
    compute_fn: Callable


class Clocked(Operator):
    """Gates a Python operator so it only fires when a clock input ticks.

    Wraps any Python [`Operator`][tradingflow.Operator] and **prepends** a
    clock handle to its inputs (position 0), mirroring the Rust
    [`Clocked<O>`][tradingflow.Clocked] layout.  On each flush, if the
    clock did not produce, the operator returns immediately without
    computing.  When the clock ticks, the inner operator's ``compute`` is
    called with the data inputs (``inputs[1:]``) and the correspondingly
    sliced ``produced[1:]`` tuple — symmetric slicing on both arguments.

    Parameters
    ----------
    clock
        Clock source handle (position 0).  The operator fires only when
        this ticks.
    inner
        The operator to wrap.

    Examples
    --------
    ```python
    monthly_clock = sc.add_source(MonthlyClock(start, end))
    universe = sc.add_operator(
        Clocked(
            monthly_clock,
            Map(market_cap, lambda m: weights(m), shape=(n,), dtype=float),
        )
    )
    ```
    """

    __slots__ = ("_inner",)

    def __init__(self, clock: Handle, inner: Operator) -> None:
        self._inner = inner
        super().__init__(
            inputs=(clock, *inner.inputs),
            kind=inner.kind,
            dtype=inner.dtype,
            shape=inner.shape,
            name=f"Clocked({getattr(inner, 'name', type(inner).__name__)})",
        )

    def init(self, inputs: tuple, timestamp: int) -> ClockedState:
        # inputs[0] is the clock (unit, ignored); inputs[1:] are data inputs.
        inner_state = self._inner.init(inputs[1:], timestamp)
        return ClockedState(
            inner_state=inner_state,
            compute_fn=type(self._inner).compute,
        )

    @staticmethod
    def compute(
        state: ClockedState,
        inputs: tuple,
        output: Any,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Clock is at position 0. Only run inner when it ticks.
        if not produced[0]:
            return False
        # Symmetric slicing: inner sees its own inputs at positions 0..
        # and the corresponding produced bits at positions 0..
        return state.compute_fn(
            state.inner_state,
            inputs[1:],
            output,
            timestamp,
            produced[1:],
        )
