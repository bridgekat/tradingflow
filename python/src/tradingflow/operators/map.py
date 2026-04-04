"""Map operators -- apply functions to transform array values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..operator import Operator, Notify
from ..types import Array, Handle, NodeKind


@dataclass
class MapState:
    """State for [`Map`]."""

    f: Callable[[np.ndarray], np.ndarray]


@dataclass
class MapInplaceState:
    """State for [`MapInplace`]."""

    f: Callable[[np.ndarray, np.ndarray], bool]


class Map[S: np.generic, T: np.generic](
    Operator[
        tuple[Handle[Array[S]]],
        Handle[Array[T]],
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

    def init(self, inputs: tuple, timestamp: int) -> MapState:
        return MapState(f=self._f)

    @staticmethod
    def compute(
        state: MapState,
        inputs: tuple,
        output: Any,
        timestamp: int,
        notify: Notify,
    ) -> bool:
        value = inputs[0].value()
        result = state.f(value)
        output.write(result)
        return True
