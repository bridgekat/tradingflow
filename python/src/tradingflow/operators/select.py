"""Select operator."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


class Select(NativeOperator):
    """Select elements by flat indices.

    Parameters
    ----------
    a
        Handle to an Array node.
    indices
        Flat indices into the input array to select.
    """

    def __init__(self, a: Handle, indices: list[int]) -> None:
        super().__init__(
            kind="select",
            inputs=(a,),
            shape=(len(indices),),
            dtype=a.dtype,
            params={"indices": indices},
        )
