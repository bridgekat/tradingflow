"""Where operator -- element-wise conditional replacement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import Array, ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class WhereState[T: np.generic]:
    """State for [`Where`]."""

    condition: Callable[[Any], bool]
    fill: float | int


class Where[T: np.generic](
    Operator[
        ArrayView[T],
        ArrayView[T],
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

    def init(self, inputs: tuple[ArrayView[T]], timestamp: int) -> WhereState[T]:
        return WhereState(condition=self._condition, fill=self._fill)

    @staticmethod
    def compute(
        state: WhereState[T],
        inputs: tuple[ArrayView[T]],
        output: ArrayView[T],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        value = inputs[0].value()
        cond_fn = np.vectorize(state.condition)
        mask = cond_fn(value)
        result = np.where(mask, value, state.fill)
        output.write(result)
        return True
