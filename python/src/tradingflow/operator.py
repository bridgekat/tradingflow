"""Operator interface — how new transformations are added to the graph.

An **operator** is a node that reads from one or more upstream nodes
and writes to its own output node.  This module defines the two
abstract bases for operators:

- [`Operator`][tradingflow.operator.Operator] — base for **Python operators**.
  Subclasses implement `compute()` (and optionally `init()`), which is
  invoked under the GIL whenever an upstream input produces.
  Convenient when the per-tick cost is dominated by something already
  in Python (CVXPY solves, scikit-learn fits, NumPy linear algebra),
  and when prototyping new ideas where the iteration speed of editing
  Python far outweighs the runtime cost of the GIL.
- [`NativeOperator`][tradingflow.operator.NativeOperator] — descriptor for
  **Rust operators**.  Subclasses are thin Python shims that carry
  enough metadata for the Rust runtime to construct and dispatch the
  native operator entirely on its own — no Python is involved on the
  hot path.

Most users won't subclass either of these directly; the built-in
operators in [`tradingflow.operators`][tradingflow.operators] cover
the common cases.  Subclass `Operator` when you need a Python-side
transformation that doesn't fit any built-in (or use the convenience
operators [`Map`][tradingflow.operators.map.Map] or
[`Apply`][tradingflow.operators.apply.Apply] for a quick lambda-style
escape hatch).  Subclass `NativeOperator` when you've added a new
operator on the Rust side and want a Python-friendly constructor for
it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .data.types import Handle, NodeKind, _to_native_node_kind

if TYPE_CHECKING:
    from tradingflow._native import NativeScenario


class Operator[*Views, Output, State](ABC):
    """Abstract base for Python-implemented operators.

    Generic parameters describe the operator's compute-time signature:

    - ***Views** — element-wise view types of the upstream inputs, in
      order.  Each entry is the view class a Python operator actually
      sees at compute time, e.g. [`ArrayView[T]`][tradingflow.data.views.ArrayView],
      [`SeriesView[T]`][tradingflow.data.views.SeriesView], or `None` for clock
      (`Unit`) inputs.
    - **Output** — view type of the operator's output node.
    - **State** — mutable state object carried across invocations.

    The flat shape mirrors the bridge: Python operators always receive a
    flat `inputs: tuple[*Views]` and a flat `produced: tuple[bool, ...]`,
    regardless of any hierarchical structure on the Rust side.

    Construction-time inputs are passed positionally as
    [`Handle`][tradingflow.data.types.Handle]s via the `inputs` constructor
    argument; the runtime resolves each handle to its corresponding
    view before calling [`init`][tradingflow.operator.Operator.init] and
    [`compute`][tradingflow.operator.Operator.compute].

    Parameters
    ----------
    inputs
        Tuple of upstream handles, in the same order as `*Views`.
    kind
        Output node kind: [`NodeKind.ARRAY`][tradingflow.data.types.NodeKind] or
        [`NodeKind.SERIES`][tradingflow.data.types.NodeKind].
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
    def init(self, inputs: tuple[*Views], timestamp: int) -> State:
        """Initialize mutable state from input views and initial timestamp.

        Consumes the operator's configuration into the returned state,
        mirroring Rust's `Operator::init` which consumes `self`.

        Parameters
        ----------
        inputs
            Tuple of input views (the same views passed to `compute`).
        timestamp
            Initial timestamp in **TAI nanoseconds** (`int64` since the
            PTP epoch 1970-01-01 00:00:00 TAI — matches numpy
            `datetime64[ns]` numerically).

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
        inputs: tuple[*Views],
        output: Output,
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        """Compute the next output value.

        Write into *output* via `output.write(value)` (Array) or
        `output.push(timestamp, value)` (Series).  Modify *state*
        in-place as needed.

        Parameters
        ----------
        state
            Mutable state (from `init` or previous `compute` call).
        inputs
            Tuple of input views, one per upstream handle, in the order
            declared by `*Views`.
        output
            Output view to write results into.
        timestamp
            Current event timestamp in **TAI nanoseconds** (`int64`
            since the PTP epoch 1970-01-01 00:00:00 TAI — matches numpy
            `datetime64[ns]` numerically).
        produced
            Flat `tuple[bool, ...]` parallel to `inputs`: element `i`
            is `True` iff input `i` produced new output this flush
            cycle.  Same arity as `inputs`; access with `produced[i]`
            or destructure (`a, b = produced`).  Transformers like
            [`Clocked`][tradingflow.operators.clocked.Clocked] forward a sliced
            view (`produced[1:]`) to inner operators.

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

    def get_io_types(self) -> tuple[list[tuple[NodeKind, str]], tuple[NodeKind, str]]:
        """Return `(input_types, output_type)` for Rust TypeId validation.

        Derived directly from the upstream handles' kind/dtype and the
        operator's declared output kind/dtype — no runtime inspection
        of generic parameters.  The class-level `*Views`/`Output` are
        purely static-typing aids for pyright.
        """
        input_names = [(inp.kind, "" if inp.kind == NodeKind.UNIT else inp.dtype.name) for inp in self._inputs]
        output_name = (self._kind, "" if self._kind == NodeKind.UNIT else self._dtype.name)
        return input_names, output_name

    def _register(self, native_scenario: NativeScenario, input_indices: list[int]) -> int:
        """Register this Python operator with the native scenario.

        Polymorphic dispatch: [`Scenario.add_operator`][tradingflow.scenario.Scenario.add_operator]
        delegates to this method without branching on operator kind.
        """
        input_types, output_type = self.get_io_types()
        return native_scenario.add_py_operator(
            input_indices,
            [(_to_native_node_kind(k), d) for k, d in input_types],
            (_to_native_node_kind(output_type[0]), output_type[1]),
            list(self._shape),
            self,
        )


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
        Output node kind: `"array"`, `"series"`, or `"unit"`.
    dtype
        Output numpy dtype.  Omit (or pass `None`) for
        [`NodeKind.UNIT`][tradingflow.data.types.NodeKind] outputs, which
        carry no value.
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
        dtype: type | np.dtype | None = None,
        shape: tuple[int, ...],
        params: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        self._native_id = native_id
        self._inputs = inputs
        self._kind = kind
        self._dtype = np.dtype(dtype) if dtype is not None else None
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
    def dtype(self) -> np.dtype | None:
        """Output numpy dtype, or `None` for Unit outputs."""
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

    def _register(self, native_scenario: NativeScenario, input_indices: list[int]) -> int:
        """Register this native operator with the native scenario."""
        return native_scenario.add_native_operator(
            self._native_id,
            self._dtype.name if self._dtype is not None else None,
            input_indices,
            list(self._shape),
            self._params,
        )
