"""Const operator factory."""

from __future__ import annotations

import numpy as np

from ..operator import NativeOperator
from ..types import Handle


def const(
    shape: tuple[int, ...],
    dtype: type | np.dtype = np.float64,
) -> NativeOperator:
    """A zero-input node holding a constant zero-filled array.

    The output is set once at init and never changes. The value can be
    mutated externally via the scenario's value access methods.

    Parameters
    ----------
    shape
        Shape of the output array.
    dtype
        NumPy dtype of the output (default ``float64``).
    """
    target = np.dtype(dtype)
    return NativeOperator(
        kind="const",
        inputs=(),
        shape=shape,
        dtype=target,
        params={"shape": list(shape)},
    )
