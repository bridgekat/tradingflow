"""Operator interface for the computation graph."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import Handle, NodeKind, node_type_to_name
from .views import Notify


class Operator[Inputs, Output, State](ABC):
    """Abstract base for Python-implemented operators.

    Generic parameters encode input/output types for both static checking
    and runtime TypeId validation:

    - **Inputs** — tuple of `Handle[NodeType]` types consumed by the operator.
    - **Output** — `Handle[NodeType]` type produced.
    - **State** — mutable state type carried across invocations.

    At registration time, `get_io_types()` extracts `(kind, dtype)` pairs
    from the generic parameters and sends them to Rust for TypeId validation.

    Parameters
    ----------
    inputs
        Tuple of upstream handles.
    kind
        Output node kind: ``"array"`` or ``"series"``.
    dtype
        NumPy dtype of the output.
    shape
        Shape of each output value element.
    name
        Optional human-readable name.
    """

    __slots__: tuple[str, ...] = ("_inputs", "_kind", "_dtype", "_shape", "_name")

    def __init__(
        self,
        inputs: tuple[Handle, ...],
        *,
        kind: NodeKind,
        dtype: type | np.dtype,
        shape: tuple[int, ...],
        name: str | None = None,
    ) -> None:
        self._inputs = inputs
        self._kind = kind
        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._name = name or type(self).__name__

    @abstractmethod
    def init(self, inputs: Inputs, timestamp: int) -> State:
        """Initialize mutable state from input views and initial timestamp.

        Consumes the operator's configuration into the returned state,
        mirroring Rust's `Operator::init` which consumes `self`.

        Parameters
        ----------
        inputs
            Tuple of input views (same views passed to `compute`).
        timestamp
            Initial timestamp (nanoseconds since epoch).

        Returns
        -------
        State
            The initial mutable state object (should contain any
            configuration needed by `compute`).
        """
        ...

    @staticmethod
    @abstractmethod
    def compute(
        state: State,
        inputs: Any,
        output: Any,
        timestamp: int,
        notify: Notify,
    ) -> bool:
        """Compute the next output value.

        Write into *output* via `output.write(value)`.  Modify *state*
        in-place as needed.

        Parameters
        ----------
        state
            Mutable state (from `init` or previous `compute` call).
        inputs
            Tuple of input views corresponding to the upstream handles.
        output
            Output view to write results into.
        timestamp
            Current event timestamp (nanoseconds since epoch).
        notify
            [`Notify`][tradingflow.Notify] context for checking which inputs
            produced new output in the current flush cycle via
            [`Notify.input_produced`][tradingflow.Notify.input_produced]
            (per-position booleans) or
            [`Notify.produced`][tradingflow.Notify.produced]
            (list of positions).

        Returns
        -------
        bool
            `True` if a value was written and downstream propagation
            should proceed.
        """
        ...

    @property
    def inputs(self) -> tuple[Handle, ...]:
        """Tuple of upstream handles."""
        return self._inputs

    @property
    def kind(self) -> NodeKind:
        """Output node kind."""
        return self._kind

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the output."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of each output value element."""
        return self._shape

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name

    @property
    def is_clock_triggerable(self) -> bool:
        """Whether this operator can be gated by a clock trigger.

        Operators that use message-passing semantics (relying on
        ``Notify.produced()`` to track which inputs produced)
        should return ``False`` — they must be triggered by their data
        inputs so they never miss a message.

        The default is ``True``.
        """
        return True

    def get_io_types(self) -> tuple[list[tuple[NodeKind, str]], tuple[NodeKind, str]]:
        """Return `(input_types, output_type)` for Rust TypeId validation.

        Tries to extract from generic type parameters first. Falls back to
        deriving types from the input handles' dtype metadata when generics
        contain unresolved TypeVars.
        """
        result = _try_extract_io_types(type(self))
        if result is not None:
            return result
        input_names = [(inp.kind, str(inp.dtype)) for inp in self._inputs]
        output_name = (self._kind, str(self._dtype))
        return input_names, output_name


def _try_extract_io_types(cls: type) -> tuple[list[tuple[NodeKind, str]], tuple[NodeKind, str]] | None:
    """Try to extract I/O types from generic parameters.

    Returns `None` if generics contain unresolved TypeVars.
    """
    for base in getattr(cls, "__orig_bases__", ()):
        origin = typing.get_origin(base)
        if origin is Operator:
            args = typing.get_args(base)
            if len(args) >= 2:
                try:
                    inputs_tp, output_tp = args[0], args[1]
                    input_handle_types = typing.get_args(inputs_tp)
                    input_names = []
                    for ht in input_handle_types:
                        ht_args = typing.get_args(ht)
                        if ht_args:
                            input_names.append(node_type_to_name(ht_args[0]))
                        else:
                            return None
                    out_args = typing.get_args(output_tp)
                    if out_args:
                        output_name = node_type_to_name(out_args[0])
                    else:
                        return None
                    return input_names, output_name
                except TypeError:
                    return None
    return None


class NativeOperator:
    """Descriptor for a Rust-implemented operator.

    Carries `native_id` + `params` — dispatched entirely on the native side.

    Parameters
    ----------
    native_id
        Operator native dispatch string (e.g. `"add"`, `"concat"`).
    inputs
        Tuple of upstream handles.
    kind
        Output node kind: ``"array"`` or ``"series"``.
    dtype
        Output numpy dtype.
    shape
        Output element shape.
    params
        Operator-specific parameters.
    name
        Optional human-readable name.
    """

    __slots__ = ("_native_id", "_inputs", "_kind", "_dtype", "_shape", "_params", "_name")

    def __init__(
        self,
        native_id: str,
        inputs: tuple[Handle, ...],
        *,
        kind: NodeKind,
        dtype: type | np.dtype,
        shape: tuple[int, ...],
        params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._native_id = native_id
        self._inputs = inputs
        self._kind = kind
        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._params = params or {}
        self._name = name or native_id

    @property
    def native_id(self) -> str:
        """Operator native dispatch string."""
        return self._native_id

    @property
    def inputs(self) -> tuple[Handle, ...]:
        """Tuple of upstream handles."""
        return self._inputs

    @property
    def kind(self) -> NodeKind:
        """Output node kind."""
        return self._kind

    @property
    def dtype(self) -> np.dtype:
        """Output numpy dtype."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Output element shape."""
        return self._shape

    @property
    def params(self) -> dict[str, Any]:
        """Operator-specific parameters."""
        return self._params

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name
