"""Filter operator -- predicate-gated passthrough."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..operator import Operator, Notify
from ..types import Array, Handle


class Filter[T: np.generic](
    Operator[
        tuple[Handle[Array[T]]],
        Handle[Array[T]],
        None,
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
            shape=input.shape,
            dtype=input.dtype,
            name=name,
        )

    def init_state(self) -> None:
        return None

    def compute(
        self,
        timestamp: int,
        inputs: tuple,
        output: Any,
        state: Any,
        notify: Notify,
    ) -> tuple[bool, Any]:
        value = inputs[0].value()
        if self._predicate(value):
            output.write(value)
            return True, state
        return False, state
