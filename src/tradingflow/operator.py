"""Core interface for computations which generate observable values.

[`Operator`][tradingflow.Operator] is the abstract base for all operators.

[`NativeOperator`][tradingflow.NativeOperator] is a concrete subclass backed
by the Rust native extension.  It holds an opaque handle that the
[`Scenario`][tradingflow.Scenario] registers directly with the Rust runtime,
bypassing Python ``compute()`` entirely during execution.  A Python fallback
``compute()`` is provided for standalone unit-test usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, override

import numpy as np
from numpy.typing import ArrayLike

from .series import AnyShape, Series

from tradingflow_native import NativeOpHandle


class Operator[Inputs, Shape: AnyShape, T: np.generic, State](ABC):
    """Abstract base class for operators that compute derived values.

    `Operator[Inputs, Shape, T, State]` is parameterised by four type
    variables:

    * **Inputs** – a tuple of input objects ([`Observable`][tradingflow.Observable]
      or [`Series`][tradingflow.Series]) consumed by the operator.  Operators
      that only need the latest value should accept `Observable` inputs;
      operators that need historical data should accept `Series` inputs.
    * **Shape** – the element shape of the output.
    * **T** – the NumPy scalar type of the output.
    * **State** – the type of the mutable state object carried across
      invocations.

    An `Operator` is a pure **specification**: it declares its inputs,
    output shape/dtype, initial state, and computation logic, but holds no
    mutable runtime state itself.  [`Scenario`][tradingflow.Scenario] owns the
    output [`Observable`][tradingflow.Observable] and the current state, and
    calls [`compute`][.compute] at each timestamp.

    Subclass `Operator` and override [`init_state`][.init_state] and [`compute`][.compute]
    to define custom computation logic.

    Parameters
    ----------
    inputs
        Tuple of input objects (Observable or Series) whose data feeds into
        the operator.
    shape
        Shape of each output value element.  Use `()` for scalars.
    dtype
        NumPy dtype for the output value buffer (e.g. `np.float64`).
    name
        Optional human-readable name used in diagnostics and error messages;
        defaults to the class name.
    """

    __slots__: tuple[str, ...] = ("_inputs", "_shape", "_dtype", "_name")

    _inputs: Inputs
    _shape: Shape
    _dtype: np.dtype[T]
    _name: str

    def __init__(
        self,
        inputs: Inputs,
        shape: Shape,
        dtype: type[T] | np.dtype[T],
        *,
        name: str | None = None,
    ) -> None:
        self._inputs = inputs
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._name = name or type(self).__name__

    @abstractmethod
    def init_state(self) -> State:
        """Returns the initial computation state for this operator.

        Called once by [`Scenario`][tradingflow.Scenario] when the operator is
        registered.  The returned value is passed as *state* to [`compute`][..compute]
        on the first invocation, and updated state is used on subsequent calls.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, timestamp: np.datetime64, inputs: Inputs, state: State) -> tuple[ArrayLike | None, State]:
        """Computes the next output value from inputs and hidden state.

        Parameters
        ----------
        timestamp
            The current timestamp.
        inputs
            The input tuple ([`Observable`][tradingflow.Observable] or
            [`Series`][tradingflow.Series] objects).
        state
            The current internal state of the operator.

        Returns
        -------
        tuple[ArrayLike | None, State]
            A `(value, new_state)` pair.  Return `None` as the value to
            skip appending an output entry for this timestamp.
        """
        raise NotImplementedError

    @property
    def inputs(self) -> Inputs:
        """The input tuple (read-only)."""
        return self._inputs

    @property
    def shape(self) -> Shape:
        """The element shape of the output."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[T]:
        """The NumPy dtype of the output."""
        return self._dtype

    @property
    def name(self) -> str:
        """Human-readable name for debugging."""
        return self._name


class NativeOperator(Operator[Any, Any, Any, None]):
    """Operator backed by the Rust native extension.

    Stores a factory that creates a fresh :class:`NativeOpHandle` on demand,
    so the same operator object can be registered in multiple scenarios or
    registered twice in the same scenario.

    During :meth:`Scenario.run`, the handle is consumed by the Rust runtime
    and the operator executes entirely in Rust — no Python ``compute()``
    callback.

    Parameters
    ----------
    handle_factory
        Callable returning a fresh :class:`NativeOpHandle`.
    inputs
        Tuple of input observables / series.
    shape
        Element shape of the output.
    dtype
        NumPy dtype of the output.
    name
        Optional human-readable name.
    """

    __slots__ = ("_handle_factory",)

    _handle_factory: Callable[[], NativeOpHandle]

    def __init__(
        self,
        handle_factory: Callable[[], NativeOpHandle],
        inputs: tuple,
        shape: tuple[int, ...],
        dtype: type | np.dtype,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(inputs, shape, dtype, name=name)
        self._handle_factory = handle_factory

    def create_handle(self) -> NativeOpHandle:
        """Create a fresh opaque Rust operator handle."""
        return self._handle_factory()

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(self, timestamp: np.datetime64, inputs: Any, state: None) -> tuple[ArrayLike | None, None]:
        raise NotImplementedError(
            "NativeOperator.compute() cannot be called directly; "
            "use Scenario.run() which delegates to the Rust runtime."
        )
