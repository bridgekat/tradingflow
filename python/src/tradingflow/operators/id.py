"""Identity operator."""

from __future__ import annotations

from .. import Handle, NativeOperator, NodeKind


class Id(NativeOperator):
    """Identity passthrough: clones input to output unchanged.

    Useful as a trigger-gated passthrough when combined with a clock.

    Parameters
    ----------
    a
        Handle to an Array node.
    """

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="id", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)
