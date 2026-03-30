"""Cast operator."""

from __future__ import annotations

import numpy as np

from ..operator import NativeOperator
from ..types import Handle


class Cast(NativeOperator):
    """Element-wise type conversion: ``out[i] = input[i] as dtype``.

    Uses truncating/saturating semantics (equivalent to Rust ``as``).

    Parameters
    ----------
    a
        Handle to an Array node.
    dtype
        Target numpy dtype.
    """

    def __init__(self, a: Handle, dtype: type | np.dtype) -> None:
        target = np.dtype(dtype)
        super().__init__(
            kind="cast",
            inputs=(a,),
            shape=a.shape,
            dtype=target,
            params={"from_dtype": str(a.dtype)},
        )
