"""Filter operator -- predicate-gated passthrough."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..operator import Operator, Notify
from ..types import Array, Handle, NodeKind


@dataclass(slots=True)
class FilterState[T: np.generic]:
    """State for [`Filter`]."""

    predicate: Callable[[np.ndarray], bool]


class Filter[T: np.generic](
    Operator[
        tuple[Handle[Array[T]]],
        Handle[Array[T]],
        FilterState[T],
    ]
):
    """Passes or drops the entire element based on a predicate.

    When the predicate returns `False`, the operator produces no output
    for that timestamp, halting downstream propagation.

    Parameters
    ----------
    input
        Upstream handle.
    predicate
        Callable receiving a numpy array (the current value) and returning
        `True` to pass it through, `False` to drop.
    name
        Optional human-readable name.
    """

    __slots__ = ("_predicate",)

    def __init__(
        self,
        input: Handle[Array[T]],
        predicate: Callable[[np.ndarray], bool],
        *,
        name: str | None = None,
    ) -> None:
        self._predicate = predicate
        super().__init__(
            inputs=(input,),
            kind=NodeKind.ARRAY,
            dtype=input.dtype,
            shape=input.shape,
            name=name,
        )

    def init(self, inputs: tuple, timestamp: int) -> FilterState[T]:
        return FilterState(predicate=self._predicate)

    @staticmethod
    def compute(
        state: FilterState,
        inputs: tuple,
        output: Any,
        timestamp: int,
        notify: Notify,
    ) -> bool:
        value = inputs[0].value()
        if state.predicate(value):
            output.write(value)
            return True
        return False
