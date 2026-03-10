"""Core time series implementation backed by NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast, overload

import numpy as np


type AnyShape = tuple[int, ...]
type Array[Shape: AnyShape, T: np.generic] = np.ndarray[Shape, np.dtype[T]]

_INITIAL_CAPACITY = 16


class Series[Shape: AnyShape, T: np.generic]:
    """A time series backed by NumPy arrays.

    ``Series[Shape, T]`` stores ``(timestamp, value)`` pairs with strictly
    increasing ``np.datetime64[ns]`` timestamps.  Both timestamps and
    values are stored in contiguous NumPy arrays:

    * **Timestamps** are stored in a 1-dimensional ``ndarray`` with
      nanosecond resolution.
    * **Values** are stored in an (N+1)-dimensional ``ndarray`` where N is the
      number of dimensions of each element.  All elements must share the
      same shape and underlying type.  Scalars are always packed in
      0-dimensional arrays (shape ``()``) to keep the API uniform.

    Internal buffers are allocated at construction time with pre-determined
    dtypes and use a doubling strategy for amortized O(1) appends.

    Parameters
    ----------
    shape
        Shape of each value element.  Use ``()`` for scalar values.
    dtype
        NumPy dtype for the value buffer (e.g. ``np.float64``).

    Invariants
    ----------
    * Timestamps are ``np.datetime64[ns]`` and strictly increasing;
      :meth:`append` enforces this.
    * Once stored, entries are never modified or removed.
    * All elements have the same shape and underlying type.

    Element access
    --------------
    * ``s[i]`` – the *i*-th value as an ``ndarray`` (supports negative
      indexing).  Scalar series return a 0-dimensional array.
    * ``s[i:j]`` – slice by integer range, returning a read-only view
      :class:`Series` without copying buffers.
    * ``s.at(ts)`` – as-of lookup: latest value at or before *ts*.
    * ``s.to(ts)`` – sub-series up to and including *ts*.
    * ``s.between(lo, hi)`` – sub-series with timestamps in ``[lo, hi]``.

    Properties
    ----------
    * :attr:`shape` – element shape of each stored value.
    * :attr:`dtype` – NumPy dtype of the value buffer.
    * :attr:`index` – read-only 1-dimensional ``ndarray`` view of timestamps.
    * :attr:`values` – read-only (N+1)-dimensional ``ndarray`` view of values.
    * :attr:`last` – most recent value as an ``ndarray``, or ``None``.
    """

    __slots__: tuple[str, ...] = ("_shape", "_length", "_index", "_values")

    _shape: Shape
    _length: np.intp
    _index: Array[tuple[int], np.datetime64]
    _values: Array[tuple[int, *Shape], T]

    def __init__(
        self,
        shape: Shape,
        dtype: type[T] | np.dtype[T],
        *,
        _length: np.intp = np.intp(0),
        _index: Array[tuple[int], np.datetime64] | None = None,
        _values: Array[tuple[int, *Shape], T] | None = None,
    ) -> None:
        if _index is not None and _values is not None:
            self._shape = cast(Shape, _values.shape[1:])
            self._length = _length
            self._index = _index
            self._values = _values
        else:
            self._shape = shape
            self._length = np.intp(0)
            self._index = np.empty(_INITIAL_CAPACITY, dtype=np.dtype("datetime64[ns]"))
            self._values = np.empty((_INITIAL_CAPACITY, *shape), dtype=dtype)  # type: ignore[assignment]

    def __len__(self) -> int:
        return int(self._length)

    def __bool__(self) -> bool:
        return bool(self._length)

    def __iter__(self) -> Iterator[tuple[np.datetime64, Array[Shape, T]]]:
        for i in np.arange(self._length):
            yield self._index[i], self._values[i, ...]  # type: ignore[return-value]

    @overload
    def __getitem__(self, key: int) -> Array[Shape, T]: ...
    @overload
    def __getitem__(self, key: slice[int, int, int]) -> Series[Shape, T]: ...

    def __getitem__(self, key: int | slice[int, int, int]) -> Array[Shape, T] | Series[Shape, T]:
        if isinstance(key, int):
            i = np.intp(key)
            if i < 0:
                i += self._length
            if not 0 <= i < self._length:
                raise IndexError(f"index {key} is out of bounds for Series of length {self._length}")
            return self._values[i, ...]  # type: ignore[return-value]
        else:
            return self._readonly_slice(key)

    @property
    def shape(self) -> Shape:
        """Element shape of each stored value."""
        return self._shape

    @property
    def dtype(self) -> np.dtype[T]:
        """NumPy dtype of the value buffer."""
        return self._values.dtype

    @property
    def index(self) -> Array[tuple[int], np.datetime64]:
        """Timestamp index as a read-only 1-D ``ndarray``."""
        view = self._index[: self._length]
        view.flags.writeable = False
        return view

    @property
    def values(self) -> Array[tuple[int, *Shape], T]:
        """Values as a read-only ``ndarray``."""
        view = self._values[: self._length]
        view.flags.writeable = False
        return view

    @property
    def last(self) -> Array[Shape, T]:
        """Most recent value as an ``ndarray``.

        For scalar series (shape ``()``), returns a 0-dimensional array.
        """
        return self._values[self._length - 1, ...]  # type: ignore[return-value]

    def at(self, timestamp: np.datetime64) -> Array[Shape, T] | None:
        """Returns the latest value at or before *timestamp* (as-of lookup).

        Returns ``None`` if no entry exists at or before *timestamp*.
        For scalar series (shape ``()``), returns a 0-dimensional array.
        """
        if self._length == 0:
            return None
        i = np.searchsorted(self._index[: self._length], timestamp, side="right") - 1
        if i < 0:
            return None
        return self._values[i, ...]  # type: ignore[return-value]

    def to(self, timestamp: np.datetime64, inclusive: bool = True) -> Series[Shape, T]:
        """Returns a read-only view with timestamps at or before *timestamp*."""
        if self._length == 0:
            return self._readonly_empty()
        r = np.searchsorted(self._index[: self._length], timestamp, side=("right" if inclusive else "left"))
        if r == 0:
            return self._readonly_empty()
        return self._readonly_slice(slice(None, int(r)))

    def between(
        self, start: np.datetime64, end: np.datetime64, left_inclusive: bool = True, right_inclusive: bool = True
    ) -> Series[Shape, T]:
        """Returns a read-only view with timestamps in ``[start, end]``."""
        if self._length == 0:
            return self._readonly_empty()
        ts = self._index[: self._length]
        l = np.searchsorted(ts, start, side=("left" if left_inclusive else "right"))
        r = np.searchsorted(ts, end, side=("right" if right_inclusive else "left"))
        if l >= r:
            return self._readonly_empty()
        return self._readonly_slice(slice(int(l), int(r)))

    def append(self, timestamp: np.datetime64, value: Array[Shape, T]) -> None:
        """Appends a ``(timestamp, value)`` pair.

        The value must have the same element shape and a dtype compatible
        with the value buffer (dtype casting follows NumPy assignment rules).

        Raises :class:`ValueError` if *timestamp* is not strictly greater than
        the current last timestamp or if the value shape doesn't match.
        """

        # Monotonicity check.
        if self._length > 0:
            if timestamp <= self._index[self._length - 1]:
                raise ValueError(
                    f"timestamp {timestamp!r} is not greater than the last "
                    f"timestamp {self._index[self._length - 1]!r}"
                )

        # Shape check.
        if value.shape != self._shape:
            raise ValueError(f"value shape {value.shape} does not match the expected " f"shape {self._shape}")

        # Grow if at capacity.
        if self._length >= len(self._index):
            self._grow()

        self._index[self._length] = timestamp
        self._values[self._length] = value
        self._length += 1

    def append_unchecked(self, timestamp: np.datetime64, value: Array[Shape, T]) -> None:
        """Append without monotonicity or shape checks.

        For internal use by callers that already guarantee ordering and shape
        correctness (e.g. :class:`~tradingflow.scenario._ScenarioState`).
        """
        if self._length >= len(self._index):
            self._grow()

        self._index[self._length] = timestamp
        self._values[self._length] = value
        self._length += 1

    def _readonly_empty(self) -> Series[Shape, T]:
        """Returns an empty :class:`Series` with the same value dtype and element shape."""
        index = np.empty(0, dtype=self._index.dtype)
        values = np.empty((0, *self._shape), dtype=self._values.dtype)
        return Series(self._shape, self.dtype, _index=index, _values=values, _length=0)  # type: ignore[arg-type]

    def _readonly_slice(self, key: slice) -> Series[Shape, T]:
        """Returns a read-only sliced :class:`Series` view without copying."""
        index = self._index[: self._length][key]
        index.flags.writeable = False
        values = self._values[: self._length][key]
        values.flags.writeable = False
        return Series(self._shape, self.dtype, _index=index, _values=values, _length=len(index))  # type: ignore[arg-type]

    def _grow(self) -> None:
        """Doubles the capacity of internal buffers."""
        new_cap = len(self._index) * 2

        new_index = np.empty(new_cap, dtype=self._index.dtype)
        new_index[: self._length] = self._index[: self._length]
        self._index = new_index

        new_values = np.empty((new_cap, *self._shape), dtype=self._values.dtype)
        new_values[: self._length] = self._values[: self._length]
        self._values = new_values  # type: ignore[assignment]
