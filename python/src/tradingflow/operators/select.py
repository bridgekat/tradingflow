"""Select operator factory."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


def select(a: Handle, indices: list[int]) -> NativeOperator:
    """Select elements by flat indices.

    Parameters
    ----------
    a
        Handle to an Array node.
    indices
        Flat indices into the input array to select.
    """
    return NativeOperator(
        kind="select",
        inputs=(a,),
        shape=(len(indices),),
        dtype=a.dtype,
        params={"indices": indices},
    )
