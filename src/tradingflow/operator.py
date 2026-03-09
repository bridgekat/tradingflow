"""Core interface for computations which generate values into time series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .series import AnyShape, Series


class Operator[Inputs: tuple[Series[Any, Any], ...], Shape: AnyShape, T: np.generic, State](ABC):
    """Abstract base class for operators that compute derived time series.

    ``Operator[Inputs, Shape, T, State]`` is parameterised by four type
    variables:

    * **Inputs** – a tuple of :class:`Series` types consumed by the operator.
    * **Shape** – the element shape of the output series.
    * **T** – the NumPy scalar type of the output series.
    * **State** – the type of the mutable state object carried across
      invocations.

    An ``Operator`` is a pure **specification**: it declares its inputs,
    output shape/dtype, initial state, and computation logic, but holds no
    mutable runtime state itself.  :class:`~src.scenario.Scenario` owns the
    output :class:`~src.series.Series` and the current state, and calls
    :meth:`compute` at each timestamp.

    Subclass ``Operator`` and override :meth:`init_state` and :meth:`compute`
    to define custom computation logic.

    Parameters
    ----------
    inputs
        Tuple of input series whose data feeds into the operator.
    shape
        Shape of each output value element.  Use ``()`` for scalars.
    dtype
        NumPy dtype for the output value buffer (e.g. ``np.float64``).
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

        Called once by :class:`~src.scenario.Scenario` when the operator is
        registered.  The returned value is passed as *state* to :meth:`compute`
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
            One :class:`Series` per input, each sliced up to *timestamp*.
        state
            The current internal state of the operator.

        Returns
        -------
        tuple[ArrayLike | None, State]
            A ``(value, new_state)`` pair.  Return ``None`` as the value to
            skip appending an output entry for this timestamp.
        """
        raise NotImplementedError

    @property
    def inputs(self) -> Inputs:
        """The input series tuple (read-only)."""
        return self._inputs

    @property
    def shape(self) -> Shape:
        """The element shape of the output series."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[T]:
        """The NumPy dtype of the output series."""
        return self._dtype

    @property
    def name(self) -> str:
        """Human-readable name for debugging."""
        return self._name
