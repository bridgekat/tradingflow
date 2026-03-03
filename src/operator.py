"""Operator base class that computes derived time series from input series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

from .series import AnyShape, Array, Series


class Operator[Inputs: tuple[Series[Any, Any], ...], Shape: AnyShape, T: np.generic, State](ABC):
    """Abstract base for operators that compute derived time series.

    ``Operator[Inputs, Shape, T, State]`` is parameterised by four type
    variables:

    * **Inputs** – a tuple of :class:`Series` types consumed by the operator.
    * **Shape** – the element shape of the output series.
    * **T** – the NumPy scalar type of the output series.
    * **State** – the type of the mutable state object carried across
      invocations.

    Subclass ``Operator`` and override :meth:`compute` to define custom
    computation logic.  Calling :meth:`update` at a given timestamp
    slices every input series up to that timestamp (via
    :meth:`Series.to`) and passes the slices, together with the
    current *state*, to :meth:`compute`.  If :meth:`compute` returns a
    value, it is converted to an ``ndarray`` and appended to
    :attr:`output`; if it returns ``None``, no output entry is written.

    This cleanly separates *data* (:class:`Series`) from *computation*
    (:class:`Operator`).  The output series can be used anywhere a
    :class:`Series` is expected, including as input to other operators.

    Parameters (``__init__``)
    -------------------------
    inputs
        Tuple of input series whose data feeds into the operator.
    shape
        Shape of each output value element.  Use ``()`` for scalars.
    dtype
        NumPy dtype for the output value buffer (e.g. ``np.float64``).
    state
        Initial mutable state carried across invocations.
    """

    __slots__: tuple[str, ...] = ("_inputs", "_output", "_state")

    _inputs: Inputs
    _output: Series[Shape, T]
    _state: State

    # -- Creation ------------------------------------------------------------

    def __init__(self, inputs: Inputs, shape: Shape, dtype: type[T] | np.dtype[T], state: State) -> None:
        self._inputs = inputs
        self._output = Series(shape, dtype)
        self._state = state

    # -- Virtual method ------------------------------------------------------

    @abstractmethod
    def compute(self, timestamp: np.datetime64, inputs: Inputs, state: State) -> ArrayLike | None:
        """Computes the output value at *timestamp*.

        Subclasses **must** override this method.  It receives the
        timestamp and the input series sliced up to that timestamp
        (as-of semantics).  Returning ``None`` indicates there is no
        output for this timestamp.

        Parameters
        ----------
        timestamp
            The ``np.datetime64[ns]`` timestamp being computed.
        inputs
            One :class:`Series` per input, each sliced up to *timestamp*.
        state
            The current internal state of the operator.

        Returns
        -------
        ArrayLike | None
            The output value, compatible with :func:`numpy.asarray`, or
            ``None`` to skip appending an output entry.
        """
        raise NotImplementedError(f"{type(self).__name__} must override compute()")

    # -- Core update mechanism -----------------------------------------------

    def update(self, timestamp: np.datetime64) -> None:
        """Computes and conditionally appends the output value at *timestamp*.

        Each input series is sliced up to *timestamp* via
        :meth:`Series.to`, then :meth:`compute` is called with the
        timestamp and the sliced inputs.

        If :meth:`compute` returns ``None``, nothing is appended.

        Raises :class:`ValueError` if appending a non-``None`` value at
        *timestamp* violates the strict monotonicity invariant of the
        output series.
        """
        slices = cast(Inputs, tuple(input.to(timestamp) for input in self._inputs))
        value = self.compute(timestamp, slices, self._state)
        if value is not None:
            arr = np.asarray(value, dtype=self._output.dtype)
            if arr.shape != self._output.shape:
                raise ValueError(f"Output value shape {arr.shape} does not match expected shape {self._output.shape}")
            self._output.append(timestamp, cast(Array[Shape, T], arr))

    # -- Accessors -----------------------------------------------------------

    @property
    def inputs(self) -> Inputs:
        """The input series tuple (read-only)."""
        return self._inputs

    @property
    def output(self) -> Series[Shape, T]:
        """The output series containing computed values (read-only)."""
        return self._output

    @property
    def state(self) -> State:
        """The current internal state of the operator (read-only)."""
        return self._state
