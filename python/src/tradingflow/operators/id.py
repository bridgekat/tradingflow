"""Identity operator factory."""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


def id(a: Handle) -> NativeOperator:
    """Identity passthrough: clones input to output unchanged.

    Useful as a trigger-gated passthrough when combined with a clock.

    Parameters
    ----------
    a
        Handle to an Array node.
    """
    return NativeOperator(kind="id", inputs=(a,), shape=a.shape, dtype=a.dtype)
