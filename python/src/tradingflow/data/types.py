"""Node type markers, `NodeKind` enum, and `Handle` for the computation graph."""

from __future__ import annotations

import enum
import typing

import numpy as np

from tradingflow._native import NativeNodeKind


class NodeKind(enum.Enum):
    """Kind of a graph node's value.

    Pure-Python mirror of the Rust `NativeNodeKind` PyO3 enum (from the
    private `tradingflow._native` extension module).  The Python-facing
    API uses [`NodeKind`][tradingflow.data.types.NodeKind] so type
    checkers (which cannot read PyO3-generated classes) see a normal
    Python enum; a private helper converts to the PyO3 variant at the
    FFI boundary.
    """

    ARRAY = "array"
    SERIES = "series"
    UNIT = "unit"


def _to_native_node_kind(kind: NodeKind) -> NativeNodeKind:
    """Convert a Python [`NodeKind`][tradingflow.data.types.NodeKind] to the Rust `NativeNodeKind` PyO3 enum.

    Used by [`Scenario`][tradingflow.scenario.Scenario] when registering nodes; the
    PyO3 enum travels across the FFI boundary in place of a string tag.
    """
    match kind:
        case NodeKind.ARRAY:
            return NativeNodeKind.Array
        case NodeKind.SERIES:
            return NativeNodeKind.Series
        case NodeKind.UNIT:
            return NativeNodeKind.Unit


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


class Unit:
    """Marker for a Rust `()` (unit) node value type.

    Used for clock sources and operators that carry no data — only a
    trigger signal.  Not instantiated; used only as a type parameter.
    """

    pass


_SCALAR_NAMES: dict[type, str] = {
    np.bool_: "bool",
    np.int8: "int8",
    np.int16: "int16",
    np.int32: "int32",
    np.int64: "int64",
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.uint32: "uint32",
    np.uint64: "uint64",
    np.float32: "float32",
    np.float64: "float64",
}


def node_type_to_name(tp: type) -> tuple[NodeKind, str]:
    """Map a node type marker to `(kind, dtype)` for Rust TypeId resolution.

    Parameters
    ----------
    tp
        A parameterized type marker, e.g. `Array[np.float64]`.

    Returns
    -------
    tuple[NodeKind, str]
        `(kind, dtype_str)`, e.g. `(NodeKind.ARRAY, "float64")`.

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
        return (NodeKind.ARRAY, dtype_str)
    if origin is Series and args:
        dtype_str = _SCALAR_NAMES.get(args[0])
        if dtype_str is None:
            raise TypeError(f"Unsupported scalar type for Series: {args[0]}")
        return (NodeKind.SERIES, dtype_str)
    if tp is Unit:
        return (NodeKind.UNIT, "")
    raise TypeError(f"Cannot resolve node type: {tp}")


class Handle[T]:
    """Typed reference to a graph node.

    `T` encodes the node's value type (e.g. `Array[np.float64]`).
    This enables static type checking: pyright can verify that an operator
    expecting `Handle[Array[np.float64]]` inputs doesn't receive a
    `Handle[Array[np.int32]]`.

    At runtime, the type parameter is not enforced — Rust validates TypeIds
    when the operator is registered.
    """

    __slots__ = ("_index", "_kind", "_dtype", "_shape")

    def __init__(
        self,
        index: int,
        kind: NodeKind,
        dtype: np.dtype,
        shape: tuple[int, ...],
    ) -> None:
        self._index = index
        self._kind = kind
        self._dtype = dtype
        self._shape = shape

    @property
    def index(self) -> int:
        """Integer index of the node in the graph."""
        return self._index

    @property
    def kind(self) -> NodeKind:
        """Node kind."""
        return self._kind

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the node's value."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of the node's value."""
        return self._shape
