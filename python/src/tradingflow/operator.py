"""Operator interface for the computation graph."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import Handle, node_type_to_name
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
    shape
        Shape of each output value element.
    dtype
        NumPy dtype of the output.
    name
        Optional human-readable name.
    """

    __slots__: tuple[str, ...] = ("_inputs", "_shape", "_dtype", "_name")

    def __init__(
        self,
        inputs: tuple[Handle, ...],
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        name: str | None = None,
    ) -> None:
        self._inputs = inputs
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._name = name or type(self).__name__

    @abstractmethod
    def init_state(self) -> State:
        """Return the initial computation state."""
        ...

    @abstractmethod
    def compute(
        self,
        timestamp: int,
        inputs: Inputs,
        output: Output,
        state: State,
        notify: Notify,
    ) -> tuple[bool, State]:
        """Compute the next output value.

        Write into *output* via `output.write(value)`.

        Parameters
        ----------
        timestamp
            Current event timestamp (nanoseconds since epoch).
        inputs
            Tuple of input views corresponding to the upstream handles.
        output
            Output view to write results into.
        state
            Mutable state carried from the previous invocation (or from
            `init_state` on the first call).
        notify
            [`Notify`][tradingflow.Notify] context for checking which inputs
            produced new output in the current flush cycle via
            [`Notify.input_produced`][tradingflow.Notify.input_produced].

        Returns
        -------
        tuple[bool, State]
            `(produced, new_state)` where *produced* is `True` if a
            value was written and downstream propagation should proceed.
        """
        ...

    @property
    def inputs(self) -> tuple[Handle, ...]:
        """Tuple of upstream handles."""
        return self._inputs

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of each output value element."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the output."""
        return self._dtype

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name

    def get_io_types(self) -> tuple[list[tuple[str, str]], tuple[str, str]]:
        """Return `(input_types, output_type)` for Rust TypeId validation.

        Tries to extract from generic type parameters first. Falls back to
        deriving types from the input handles' dtype metadata when generics
        contain unresolved TypeVars.
        """
        result = _try_extract_io_types(type(self))
        if result is not None:
            return result
        input_names = [("array", str(inp.dtype)) for inp in self._inputs]
        output_name = ("array", str(self._dtype))
        return input_names, output_name


def _try_extract_io_types(cls: type) -> tuple[list[tuple[str, str]], tuple[str, str]] | None:
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

    Carries `kind` + `params` — dispatched entirely on the native side.

    Parameters
    ----------
    kind
        Operator kind string (e.g. `"add"`, `"concat"`).
    inputs
        Tuple of upstream handles.
    shape
        Output element shape.
    dtype
        Output numpy dtype.
    params
        Operator-specific parameters.
    name
        Optional human-readable name.
    """

    __slots__ = ("_kind", "_inputs", "_shape", "_dtype", "_params", "_name")

    def __init__(
        self,
        kind: str,
        inputs: tuple[Handle, ...],
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._kind = kind
        self._inputs = inputs
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._params = params or {}
        self._name = name or kind

    @property
    def kind(self) -> str:
        """Operator kind string."""
        return self._kind

    @property
    def inputs(self) -> tuple[Handle, ...]:
        """Tuple of upstream handles."""
        return self._inputs

    @property
    def shape(self) -> tuple[int, ...]:
        """Output element shape."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Output numpy dtype."""
        return self._dtype

    @property
    def params(self) -> dict[str, Any]:
        """Operator-specific parameters."""
        return self._params

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name
