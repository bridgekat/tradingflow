"""Map operator — apply a function to transform array values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import Array, ArrayView, Handle, NodeKind, Operator


@dataclass
class MapState:
    """State for [`Map`][tradingflow.operators.map.Map]."""

    f: Callable[[np.ndarray], np.ndarray]


class Map[S: np.generic, T: np.generic](
    Operator[
        ArrayView[S],
        ArrayView[T],
        MapState,
    ]
):
    """Applies a function to the input array on each tick.

    The function receives a numpy array (the current value) and must
    return a numpy array of the declared output shape and dtype.
    Always produces output (never halts propagation).

    Parameters
    ----------
    input
        Upstream handle.
    f
        Callable receiving a numpy array and returning a numpy array.
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
        input: Handle[Array[S]],
        f: Callable[[np.ndarray], np.ndarray],
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        name: str | None = None,
    ) -> None:
        self._f = f
        super().__init__(
            inputs=(input,),
            kind=NodeKind.ARRAY,
            dtype=dtype,
            shape=shape,
            name=name,
        )

    def init(self, inputs: tuple[ArrayView[S]], timestamp: int) -> MapState:
        return MapState(f=self._f)

    @staticmethod
    def compute(
        state: MapState,
        inputs: tuple[ArrayView[S]],
        output: ArrayView[T],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        value = inputs[0].value()
        result = state.f(value)
        output.write(result)
        return True
