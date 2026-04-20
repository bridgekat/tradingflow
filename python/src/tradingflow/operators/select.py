"""Select operator."""

from __future__ import annotations

from .. import Handle, NativeOperator, NodeKind


class Select(NativeOperator):
    """Select elements along an axis.

    When *indices* is a single `int`, the selected axis is squeezed
    out of the output shape (e.g. selecting element 2 from a `(5,)`
    array yields a scalar `()`; selecting column 3 from a `(N, 5)`
    array with `axis=1` yields `(N,)`).

    When *indices* is a `list[int]`, the axis is preserved with size
    `len(indices)`.

    Parameters
    ----------
    a
        Handle to an Array node.
    indices
        A single index (`int`, squeezes the axis) or a list of
        indices (`list[int]`, preserves the axis).
    axis
        Axis to select along (default `0`).
    """

    def __init__(self, a: Handle, indices: int | list[int], *, axis: int = 0) -> None:
        if isinstance(indices, int):
            squeeze = True
            indices = [indices]
        else:
            squeeze = False

        if squeeze and len(indices) != 1:
            raise ValueError(f"squeeze requires exactly one index, got {len(indices)}")

        # Compute output shape.
        in_shape = list(a.shape)
        if not in_shape:
            out_shape = () if squeeze else (len(indices),)
        else:
            in_shape[axis] = len(indices)
            if squeeze:
                del in_shape[axis]
            out_shape = tuple(in_shape)

        params: dict = {"indices": indices}
        if axis != 0:
            params["axis"] = axis
        if squeeze:
            params["squeeze"] = True

        super().__init__(
            native_id="select",
            inputs=(a,),
            kind=NodeKind.ARRAY,
            dtype=a.dtype,
            shape=out_shape,
            params=params,
        )
