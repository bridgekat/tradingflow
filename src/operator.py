"""Operator base class that computes derived time series from input series."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .series import T, Series


S = TypeVar("S")


class Operator(Generic[S, T], ABC):
    """Abstract base for operators that compute derived time series.

    Subclass ``Operator`` and override :meth:`compute` to define custom
    computation logic.  An ``Operator`` encapsulates:

    * **inputs** – zero or more :class:`Series` whose data feeds into the
      computation.
    * **state** – a mutable object of type *S* carried across invocations.
    * **output** – a :class:`Series[T]` that stores the computed values.

    Calling :meth:`update` at a given timestamp slices every input series
    up to that timestamp (via :meth:`Series.to`) and passes the slices to
    :meth:`compute`.  If :meth:`compute` returns a value, it is appended
    to :attr:`output`; if it returns ``None``, no output entry is written.

    This cleanly separates *data* (:class:`Series`) from *computation*
    (:class:`Operator`).  The output is a fixed :class:`Series` and can be
    used anywhere a :class:`Series` is expected, including as input to
    other operators.

    Parameters
    ----------
    inputs
        Input series whose data feeds into the operator.
    state
        Mutable internal state carried across invocations.
    dtype
        NumPy dtype for the output value buffer (e.g. ``np.float64``).
    shape
        Shape of each output value element.  Defaults to ``()`` for scalars.

    Example
    -------
    >>> import numpy as np
    >>> class MovingAverage(Operator[None, np.float64]):
    ...     def __init__(self, window: int, prices: Series[np.float64]) -> None:
    ...         super().__init__([prices], None, np.float64)
    ...         self._window = window
    ...
    ...     def compute(self, timestamp, prices):
    ...         vals = prices.values[-self._window:]
    ...         return float(vals.mean())
    ...
    >>> prices: Series[np.float64] = Series(np.float64)
    >>> ma = MovingAverage(2, prices)
    >>> prices.append(np.datetime64(1, "ns"), 10.0); ma.update(np.datetime64(1, "ns"))
    >>> prices.append(np.datetime64(2, "ns"), 20.0); ma.update(np.datetime64(2, "ns"))
    >>> prices.append(np.datetime64(3, "ns"), 30.0); ma.update(np.datetime64(3, "ns"))
    >>> list(ma.output.values)
    [10.0, 15.0, 25.0]
    """

    __slots__ = ("_inputs", "_state", "_output")

    def __init__(
        self,
        inputs: list[Series[Any]],
        state: S,
        dtype: np.dtype[T],
        shape: tuple[int, ...] = (),
    ) -> None:
        self._inputs: list[Series[Any]] = list(inputs)
        self._state = state
        self._output: Series[T] = Series(dtype, shape)

    # -- Virtual method ------------------------------------------------------

    @abstractmethod
    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        """Computes the output value at *timestamp*.

        Subclasses **must** override this method.  It receives the
        timestamp and the input series sliced up to that timestamp
        (as-of semantics).  Returning ``None`` indicates there is no
        output for this timestamp.

        Parameters
        ----------
        timestamp
            The ``np.datetime64[ns]`` timestamp being computed.
        *inputs
            One :class:`Series` per input, each sliced up to *timestamp*.

        Returns
        -------
        Optional[ArrayLike]
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
        slices = [s.to(timestamp) for s in self._inputs]
        value = self.compute(timestamp, *slices)
        if value is not None:
            self._output.append(timestamp, value)

    # -- Accessors -----------------------------------------------------------

    @property
    def inputs(self) -> list[Series[Any]]:
        """The input series list (read-only)."""
        return self._inputs

    @property
    def state(self) -> S:
        """The current internal state of the operator (read-only)."""
        return self._state

    @property
    def output(self) -> Series[T]:
        """The output series containing computed values (read-only)."""
        return self._output
