"""Node type markers and Handle for the computation graph.

Type markers (`Array`, `Series`) are pure Python generic classes that encode
Rust node value types. They serve dual purpose:

1. **Static type checking** — pyright/mypy can verify input/output compatibility.
2. **Runtime extraction** — `node_type_to_name()` maps them to `(kind, dtype)`
   pairs for Rust `TypeId` validation.

`Handle[NodeType]` is a typed reference to a graph node, parameterized by
the node's value type marker.
"""

from __future__ import annotations

import typing

import numpy as np


# ---------------------------------------------------------------------------
# Node type markers
# ---------------------------------------------------------------------------


class Array[T: np.generic]:
    """Marker for a Rust `Array<T>` node value type.

    Not instantiated — used only as a generic type parameter.
    """

    pass


class Series[T: np.generic]:
    """Marker for a Rust `Series<T>` node value type.

    Not instantiated — used only as a generic type parameter.
    """

    pass


# ---------------------------------------------------------------------------
# Type-to-name mapping
# ---------------------------------------------------------------------------

_SCALAR_NAMES: dict[type, str] = {
    np.float64: "float64",
    np.float32: "float32",
    np.int64: "int64",
    np.int32: "int32",
    np.uint64: "uint64",
    np.uint32: "uint32",
    np.bool_: "bool",
}


def node_type_to_name(tp: type) -> tuple[str, str]:
    """Map a node type marker to `(kind, dtype)` for Rust TypeId resolution.

    Parameters
    ----------
    tp
        A parameterized type marker, e.g. `Array[np.float64]`.

    Returns
    -------
    tuple[str, str]
        `(kind, dtype_str)`, e.g. `("array", "float64")`.

    Raises
    ------
    TypeError
        If the type is not a recognized node type marker.
    """
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Array and args:
        dtype_str = _SCALAR_NAMES.get(args[0])
        if dtype_str is None:
            raise TypeError(f"Unsupported scalar type for Array: {args[0]}")
        return ("array", dtype_str)
    if origin is Series and args:
        dtype_str = _SCALAR_NAMES.get(args[0])
        if dtype_str is None:
            raise TypeError(f"Unsupported scalar type for Series: {args[0]}")
        return ("series", dtype_str)
    raise TypeError(f"Cannot resolve node type: {tp}")


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


class Handle[NodeType]:
    """Typed reference to a graph node.

    `NodeType` encodes the node's value type (e.g. `Array[np.float64]`).
    This enables static type checking: pyright can verify that an operator
    expecting `Handle[Array[np.float64]]` inputs doesn't receive a
    `Handle[Array[np.int32]]`.

    At runtime, the type parameter is not enforced — Rust validates TypeIds
    when the operator is registered.
    """

    __slots__ = ("_index", "_shape", "_dtype")

    def __init__(self, index: int, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self._index = index
        self._shape = shape
        self._dtype = dtype

    @property
    def index(self) -> int:
        return self._index

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype
