"""Typed Python wrappers over native array and series views."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingflow._native import NativeArrayView, NativeNotify, NativeSeriesView

from .schema import Schema
from .utils import coerce_timestamp


class ArrayView[T: np.generic]:
    """View of a Rust `Array<T>` node in the computation graph.

    All reads copy data out; all writes copy data in. No reference to
    Rust-owned memory is ever exposed to Python.

    Implements the numpy array protocol (`__array__`), so instances can
    be used directly in numpy operations: `np.log(view)`, `view + 1`, etc.

    Instances are created by the runtime during operator registration —
    users do not construct these directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: NativeArrayView) -> None:
        self._inner = inner

    # -- Core access ----------------------------------------------------------

    def value(self) -> np.ndarray:
        """Copy the array data into a new numpy array."""
        return self._inner.value()

    def write(self, value: np.ndarray) -> None:
        """Overwrite the array data from a numpy array.

        Parameters
        ----------
        value
            Array whose shape must match the view's shape.
        """
        self._inner.write(value)

    def to_numpy(self) -> np.ndarray:
        """Alias for `value`."""
        return self.value()

    # -- Properties -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of the array."""
        return self._inner.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the array."""
        return self._inner.dtype

    # -- Numpy array protocol -------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """NumPy array protocol.

        Enables `np.asarray(view)`, `np.log(view)`, etc.
        """
        arr = self.value()
        return arr if dtype is None else arr.astype(dtype)

    # -- Indexing -------------------------------------------------------------

    def __getitem__(self, key):
        """NumPy-style indexing (reads a copy)."""
        return self.value()[key]

    def __setitem__(self, key, value):
        """NumPy-style indexed assignment (read-modify-write)."""
        arr = self.value()
        arr[key] = value
        self.write(arr)

    # -- Arithmetic (return numpy arrays, NOT graph ops) ----------------------

    def __add__(self, other):
        return np.asarray(self) + np.asarray(other)

    def __radd__(self, other):
        return np.asarray(other) + np.asarray(self)

    def __sub__(self, other):
        return np.asarray(self) - np.asarray(other)

    def __rsub__(self, other):
        return np.asarray(other) - np.asarray(self)

    def __mul__(self, other):
        return np.asarray(self) * np.asarray(other)

    def __rmul__(self, other):
        return np.asarray(other) * np.asarray(self)

    def __truediv__(self, other):
        return np.asarray(self) / np.asarray(other)

    def __rtruediv__(self, other):
        return np.asarray(other) / np.asarray(self)

    def __neg__(self):
        return -np.asarray(self)

    # -- Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ArrayView(shape={self.shape}, dtype={self.dtype})"


class SeriesView[T: np.generic]:
    """View of a Rust `Series<T>` node in the computation graph.

    All reads copy data out. Series buffers can reallocate during graph
    execution; copies prevent dangling.

    Instances are created by the runtime during operator registration —
    users do not construct these directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: NativeSeriesView) -> None:
        self._inner = inner

    # -- Core access ----------------------------------------------------------

    def last(self) -> np.ndarray:
        """Copy the latest element into a new numpy array."""
        return self._inner.last()

    def values(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """Copy a slice of values into a new numpy array.

        Parameters
        ----------
        start
            Start index (inclusive). Defaults to 0.
        end
            End index (exclusive). Defaults to the series length.
        """
        return self._inner.values(start, end)

    def slice(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """Copy a slice of timestamps into a new numpy int64 array.

        Parameters
        ----------
        start
            Start index (inclusive).
        end
            End index (exclusive). Defaults to the series length.
        """
        return self._inner.slice(start, end)

    # -- New convenience methods ----------------------------------------------

    def timestamps(self, start: int = 0, end: int | None = None) -> np.ndarray:
        """Timestamps as `datetime64[ns]` array.

        Wraps `slice` with a view cast to `datetime64[ns]`.
        """
        return self.slice(start, end).view("datetime64[ns]")

    def at(self, i: int) -> np.ndarray:
        """Single element by positional index (supports negative indexing).

        Delegates to Rust `Series::at` via the bridge.
        """
        return self._inner.at(i)

    def asof(self, timestamp: np.datetime64) -> np.ndarray | None:
        """Value at or before `timestamp` (binary search).

        Delegates to Rust `Series::asof` via the bridge.

        Parameters
        ----------
        timestamp
            Timestamp to search for. Can be any `datetime64` precision; it
            will be coerced to `datetime64[ns]` internally.

        Returns
        -------
        np.ndarray or None
            The most recent element with `ts <= timestamp`, or `None`
            if no element satisfies the condition.
        """
        result = self._inner.asof(coerce_timestamp(timestamp))
        return None if result is None else result

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return `(timestamps_datetime64, values)` tuple."""
        return self.timestamps(), self.values()

    def to_dataframe(self, columns=None) -> pd.DataFrame:
        """Convert to pandas DataFrame with DatetimeIndex.

        Parameters
        ----------
        columns
            Column names. Accepts a `list[str]` or a
            [`Schema`][tradingflow.Schema].
            If `None`, uses integer column names.
        """
        import pandas as pd

        ts, vals = self.to_numpy()
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim > 2:
            vals = vals.reshape(len(ts), -1)
        if isinstance(columns, Schema):
            columns = [columns.name(i) for i in range(vals.shape[1])]
        return pd.DataFrame(vals, index=pd.DatetimeIndex(ts), columns=columns)

    # -- Indexing -------------------------------------------------------------

    def __getitem__(self, key):
        """Positional indexing: `int` for single element, `slice` for range."""
        if isinstance(key, int):
            return self.at(key)
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError("only contiguous slices supported")
            return self.values(start, stop)
        raise TypeError(f"unsupported index type: {type(key).__name__}")

    # -- Properties -----------------------------------------------------------

    def __len__(self) -> int:
        """Number of recorded elements."""
        return len(self._inner)

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of each stored value."""
        return self._inner.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of stored values."""
        return self._inner.dtype

    def __repr__(self) -> str:
        return f"SeriesView(len={len(self)}, shape={self.shape}, dtype={self.dtype})"


class Notify:
    """Notification context for Python-implemented operators.

    Provides [`input_produced`][tradingflow.Notify.input_produced] to check
    whether a specific input produced new output in the current flush cycle.
    If the operator ignores this object entirely, there is zero overhead.

    Instances are created by the runtime and passed to
    [`Operator.compute`][tradingflow.Operator.compute] — users do not construct
    these directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: NativeNotify) -> None:
        self._inner = inner

    def input_produced(self, pos: int) -> bool:
        """Whether the input at *pos* produced new output this flush cycle."""
        return self._inner.input_produced(pos)
