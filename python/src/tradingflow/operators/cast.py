"""Cast operator."""

from __future__ import annotations

import numpy as np

from ..operator import NativeOperator
from ..types import Handle, NodeKind


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
            native_id="cast",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=target,
            shape=a.shape,
            params={"from_dtype": str(a.dtype)},
        )
