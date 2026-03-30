"""Identity operator."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


class Id(NativeOperator):
    """Identity passthrough: clones input to output unchanged.

    Useful as a trigger-gated passthrough when combined with a clock.

    Parameters
    ----------
    a
        Handle to an Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="id", inputs=(a,), shape=a.shape, dtype=a.dtype)
