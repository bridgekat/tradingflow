"""Cast operator factory."""

from __future__ import annotations

import numpy as np

from ..operator import NativeOperator
from ..types import Handle


def cast(a: Handle, dtype: type | np.dtype) -> NativeOperator:
    """Element-wise type conversion: ``out[i] = input[i] as dtype``.

    Uses truncating/saturating semantics (equivalent to Rust ``as``).

    Parameters
    ----------
    a
        Handle to an Array node.
    dtype
        Target numpy dtype.
    """
    target = np.dtype(dtype)
    return NativeOperator(
        kind="cast",
        inputs=(a,),
        shape=a.shape,
        dtype=target,
        params={"from_dtype": str(a.dtype)},
    )
