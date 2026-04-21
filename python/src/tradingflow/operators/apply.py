"""Apply operator — apply a function to multiple array inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import ArrayView, Handle, NodeKind, Operator


@dataclass
class ApplyState:
    """State for [`Apply`]."""

    f: Callable[..., np.ndarray]


class Apply(
    Operator[
        *tuple[ArrayView[np.float64], ...],
        ArrayView[np.float64],
        ApplyState,
    ]
):
    """Applies a function to multiple input arrays on each tick.

    Unlike [`Map`][tradingflow.operators.map.Map] which takes a single input,
    `Apply` accepts an arbitrary tuple of upstream handles.  The function
    receives the current values of all inputs as positional `np.ndarray`
    arguments and must return a `np.ndarray` of the declared output
    shape and dtype.  Always produces output (never halts propagation).

    Parameters
    ----------
    inputs
        Tuple of upstream handles.
    f
        Callable receiving N numpy arrays and returning a numpy array.
    shape
        Output element shape.
    dtype
        Output numpy dtype.
    name
        Optional human-readable name.
    """

    __slots__ = ("_f",)

    def __init__(
        self,
        inputs: tuple[Handle, ...],
        f: Callable[..., np.ndarray],
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        name: str | None = None,
    ) -> None:
        self._f = f
        super().__init__(
            inputs=inputs,
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=shape,
            name=name,
        )

    def init(self, inputs: tuple[ArrayView[np.float64], ...], timestamp: int) -> ApplyState:
        return ApplyState(f=self._f)

    @staticmethod
    def compute(
        state: ApplyState,
        inputs: tuple[ArrayView[np.float64], ...],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        values = tuple(inp.value() for inp in inputs)
        result = state.f(*values)
        output.write(result)
        return True
