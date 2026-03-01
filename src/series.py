"""Core time series implementation backed by NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, Optional, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray


T = TypeVar("T", bound=np.generic)

_INITIAL_CAPACITY = 16


class Series(Generic[T]):
    """A time series backed by NumPy arrays.

    ``Series[T]`` stores ``(timestamp, value)`` pairs with strictly
    increasing ``np.datetime64[ns]`` timestamps.  Both timestamps and
    values are stored in contiguous NumPy arrays:

    * **Timestamps** are stored in a 1-D ``NDArray[np.datetime64]`` with
      nanosecond resolution.
    * **Values** are stored in an (N+1)-D ``NDArray[T]`` where N is the
      number of dimensions of each element.  All elements must share the
      same shape and dtype.  Arbitrary Python objects are supported via
      ``dtype=object``.

    Internal buffers are allocated at construction time with pre-determined
    dtypes and use a doubling strategy for amortised O(1) appends.

    Parameters
    ----------
    dtype
        NumPy dtype for the value buffer (e.g. ``np.float64``).
    shape
        Shape of each value element.  Defaults to ``()`` for scalar values.

    Invariants
    ----------
    * Timestamps are ``np.datetime64[ns]`` and strictly increasing;
      :meth:`append` enforces this.
    * Once stored, entries are never modified or removed.
    * All elements have the same shape and dtype.
    * Internal buffers are always allocated (never empty).

    Element access
    --------------
        * ``s[i]`` – the *i*-th ``(timestamp, value)`` pair (supports negative
            indexing).
    * ``s[i:j]`` – slice by integer range, returning a new :class:`Series`.
    * ``s.at(ts)`` – as-of lookup: latest value at or before *ts*.
    * ``s.to(ts)`` – sub-series up to and including *ts*.
    * ``s.between(lo, hi)`` – sub-series with timestamps in ``[lo, hi]``.

    Properties
    ----------
    * :attr:`timestamps` – read-only 1-D ``NDArray[np.datetime64]`` view.
    * :attr:`values` – read-only (N+1)-D ``NDArray[T]`` view.
    * :attr:`last` – most recent ``(timestamp, value)`` pair, or ``None``.
    """

    __slots__ = ("_ts", "_arr", "_length")

    def __init__(
        self,
        dtype: type[T] | np.dtype[T],
        shape: tuple[int, ...] = (),
    ) -> None:
        self._ts: NDArray[np.datetime64] = np.empty(_INITIAL_CAPACITY, dtype=np.dtype("datetime64[ns]"))
        self._arr: NDArray[T] = np.empty((_INITIAL_CAPACITY, *shape), dtype=dtype)
        self._length: int = 0

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _from_raw(ts: NDArray[np.datetime64], vals: NDArray[T], length: int) -> Series[T]:
        """Wraps pre-allocated buffers (caller transfers ownership)."""
        obj = object.__new__(Series)
        obj._ts = ts
        obj._arr = vals
        obj._length = length
        return obj

    def _empty_like(self) -> Series[T]:
        """Return an empty :class:`Series` with the same value dtype and element shape."""
        return Series(self._arr.dtype, self._arr.shape[1:])

    def _grow(self) -> None:
        """Doubles the capacity of internal buffers."""
        new_cap = len(self._ts) * 2
        elem_shape = self._arr.shape[1:]

        new_ts: NDArray[np.datetime64] = np.empty(new_cap, dtype=self._ts.dtype)
        new_ts[: self._length] = self._ts[: self._length]
        self._ts = new_ts

        new_vals: NDArray[T] = np.empty((new_cap, *elem_shape), dtype=self._arr.dtype)
        new_vals[: self._length] = self._arr[: self._length]
        self._arr = new_vals

    # -- Size & truthiness ---------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return self._length > 0

    # -- Iteration -----------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[np.datetime64, T]]:
        for i in range(self._length):
            yield (self._ts[i], self._arr[i])

    # -- Element access ------------------------------------------------------

    @overload
    def __getitem__(self, key: int) -> tuple[np.datetime64, T]: ...
    @overload
    def __getitem__(self, key: slice) -> Series[T]: ...

    def __getitem__(self, key: int | slice) -> tuple[np.datetime64, T] | Series[T]:
        if isinstance(key, int):
            if key < 0:
                key += self._length
            if not 0 <= key < self._length:
                raise IndexError("Series index out of range")
            return (self._ts[key], self._arr[key])

        else:
            ts_slice = self._ts[: self._length][key].copy()
            val_slice = self._arr[: self._length][key].copy()
            return Series._from_raw(ts_slice, val_slice, len(ts_slice))

    # -- Properties ----------------------------------------------------------

    @property
    def timestamps(self) -> NDArray[np.datetime64]:
        """Timestamps as a read-only 1-D ``NDArray[np.datetime64]``."""
        view = self._ts[: self._length]
        view.flags.writeable = False
        return view

    @property
    def values(self) -> NDArray[T]:
        """Values as a read-only ``NDArray[T]``."""
        view = self._arr[: self._length]
        view.flags.writeable = False
        return view

    @property
    def last(self) -> Optional[tuple[np.datetime64, T]]:
        """Most recent ``(timestamp, value)`` pair, or ``None`` if empty."""
        if self._length == 0:
            return None
        return (self._ts[self._length - 1], self._arr[self._length - 1])

    # -- Timestamp-based access ----------------------------------------------

    def at(self, timestamp: np.datetime64) -> Optional[T]:
        """Return the latest value at or before *timestamp* (as-of lookup).

        Returns ``None`` if no entry exists at or before *timestamp*.
        For scalar elements the return is a NumPy scalar; for N-D elements
        it is an ``np.ndarray`` view.
        """
        if self._length == 0:
            return None
        i = np.searchsorted(self._ts[: self._length], timestamp, side="right") - 1
        if i < 0:
            return None
        return self._arr[i]

    def to(self, timestamp: np.datetime64) -> Series[T]:
        """Return a sub-series with timestamps at or before *timestamp*."""
        if self._length == 0:
            return self._empty_like()
        r = np.searchsorted(self._ts[: self._length], timestamp, side="right")
        if r == 0:
            return self._empty_like()
        return Series._from_raw(self._ts[:r].copy(), self._arr[:r].copy(), int(r))

    def between(self, start: np.datetime64, end: np.datetime64) -> Series[T]:
        """Return a sub-series with timestamps in ``[start, end]``."""
        if self._length == 0:
            return self._empty_like()
        ts = self._ts[: self._length]
        l = np.searchsorted(ts, start, side="left")
        r = np.searchsorted(ts, end, side="right")
        if l >= r:
            return self._empty_like()
        return Series._from_raw(self._ts[l:r].copy(), self._arr[l:r].copy(), int(r - l))

    # -- Mutation ------------------------------------------------------------

    def append(self, timestamp: np.datetime64, value: ArrayLike) -> None:
        """Append a ``(timestamp, value)`` pair.

        The value must have the same element shape and a dtype compatible
        with the value buffer (dtype casting follows NumPy assignment rules).

        Raises :class:`ValueError` if *timestamp* is not strictly greater than
        the current last timestamp or if the value shape doesn't match.
        """
        arr = np.asarray(value)

        # Monotonicity check.
        if self._length > 0:
            if timestamp <= self._ts[self._length - 1]:
                raise ValueError(
                    f"timestamp {timestamp!r} is not greater than the last " f"timestamp {self._ts[self._length - 1]!r}"
                )

        # Shape check.
        expected_shape = self._arr.shape[1:]
        if arr.shape != expected_shape:
            raise ValueError(f"value shape {arr.shape} does not match the expected " f"shape {expected_shape}")

        # Grow if at capacity.
        if self._length >= len(self._ts):
            self._grow()

        self._ts[self._length] = timestamp
        self._arr[self._length] = arr
        self._length += 1

    # -- Representation ------------------------------------------------------

    def __repr__(self) -> str:
        if self._length == 0:
            return "Series(empty)"
        elem_shape = self._arr.shape[1:]
        shape_str = f", shape={elem_shape}" if elem_shape else ""
        return (
            f"Series(length={self._length}, "
            f"first={self._ts[0]!r}, "
            f"last={self._ts[self._length - 1]!r}"
            f"{shape_str})"
        )
