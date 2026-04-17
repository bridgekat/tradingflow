"""Where operator -- element-wise conditional replacement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..operator import Operator
from ..types import Array, Handle, NodeKind


@dataclass(slots=True)
class WhereState[T: np.generic]:
    """State for [`Where`]."""

    condition: Callable[[Any], bool]
    fill: float | int


class Where[T: np.generic](
    Operator[
        tuple[Handle[Array[T]]],
        Handle[Array[T]],
        WhereState[T],
    ]
):
    """Element-wise conditional: keeps values where `condition` is `True`,
    replaces others with `fill`.

    Unlike [`Filter`][tradingflow.operators.Filter], this always produces
    output (never halts propagation).

    Parameters
    ----------
    input
        Upstream handle.
    condition
        Callable receiving a scalar value and returning `True` to keep it.
    fill
        Replacement value for elements where `condition` returns `False`.
    name
        Optional human-readable name.
    """

    __slots__ = ("_condition", "_fill")

    def __init__(
        self,
        input: Handle[Array[T]],
        condition: Callable[[Any], bool],
        fill: float | int = 0.0,
        *,
        name: str | None = None,
    ) -> None:
        self._condition = condition
        self._fill = fill
        super().__init__(
            inputs=(input,),
            kind=NodeKind.ARRAY,
            dtype=input.dtype,
            shape=input.shape,
            name=name,
        )

    def init(self, inputs: tuple, timestamp: int) -> WhereState[T]:
        return WhereState(condition=self._condition, fill=self._fill)

    @staticmethod
    def compute(
        state: WhereState,
        inputs: tuple,
        output: Any,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        value = inputs[0].value()
        cond_fn = np.vectorize(state.condition)
        mask = cond_fn(value)
        result = np.where(mask, value, state.fill)
        output.write(result)
        return True
