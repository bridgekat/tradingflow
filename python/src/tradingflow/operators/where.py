"""Where operator — element-wise conditional replacement.

Since Rust closures cannot cross FFI, this is a Python `Operator` subclass.
The condition is applied element-wise; failing elements are replaced with a
fill value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..operator import Operator
from ..types import Array, Handle


class Where[T: np.generic](
    Operator[
        tuple[Handle[Array[T]]],
        Handle[Array[T]],
        None,
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
    ) -> tuple[bool, Any]:
        value = inputs[0].value()
        cond_fn = np.vectorize(self._condition)
        mask = cond_fn(value)
        result = np.where(mask, value, self._fill)
        output.write(result)
        return True, state
