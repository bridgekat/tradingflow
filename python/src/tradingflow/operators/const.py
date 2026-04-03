"""Const operator."""

from __future__ import annotations

import numpy as np

from ..utils import ensure_contiguous
from ..operator import NativeOperator
from ..types import NodeKind


class Const(NativeOperator):
    """A zero-input node holding a constant array value.

    The output is set once at init and never recomputed. The value can
    be mutated externally via the scenario's view access methods.

    Parameters
    ----------
    value
        Initial value as a numpy array.
    """

    def __init__(self, value: np.ndarray) -> None:
        arr = ensure_contiguous(np.asarray(value))
        super().__init__(
            native_id="const",
            inputs=(),
            kind=NodeKind.ARRAY,
            dtype=arr.dtype,
            shape=arr.shape,
            params={"shape": list(arr.shape), "value": arr},
        )
