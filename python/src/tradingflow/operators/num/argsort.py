"""ArgSort operator — indices that would sort the array."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class ArgSort(NativeOperator):
    """Produces the indices that would sort a 1-D array from smallest
    to largest.  Output is `Array<uint64>` of the same length.

    NaN values are sorted to the end.

    Parameters
    ----------
    a
        Handle to a 1-D Array node with a float dtype.
    """

    def __init__(self, a: Handle) -> None:
        assert len(a.shape) == 1, "ArgSort requires a 1-D input"
        super().__init__(
            native_id="argsort",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=np.uint64,
            shape=a.shape,
            params={"input_dtype": str(a.dtype)},
        )
