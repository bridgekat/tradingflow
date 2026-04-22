"""Rank and ArgSort operators over 1-D arrays."""

from __future__ import annotations

import numpy as np

from ... import Handle, NativeOperator, NodeKind


class Rank(NativeOperator):
    """Produces the 0-based rank of each element in a 1-D array.

    The smallest element is assigned rank 0, the next smallest rank 1,
    and so on; NaN values receive the highest ranks.  Output is
    `Array<uint64>` of the same length as the input.

    Conceptually the inverse permutation of
    [`ArgSort`][tradingflow.operators.num.rank.ArgSort]:
    ``rank[argsort[i]] == i``.

    Parameters
    ----------
    a
        Handle to a 1-D Array node with a float dtype.
    """

    def __init__(self, a: Handle) -> None:
        assert len(a.shape) == 1, "Rank requires a 1-D input"
        super().__init__(
            native_id="rank",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=np.uint64,
            shape=a.shape,
            params={"input_dtype": str(a.dtype)},
        )


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
