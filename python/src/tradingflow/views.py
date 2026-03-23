"""Typed Python wrappers over the native `_ArrayView` and `_SeriesView`.

These classes provide LSP-visible type annotations, docstrings, and
autocompletion for the underlying Rust `#[pyclass]` types. All methods
delegate to the inner native view via composition.

The native views are the safety boundary — they copy data on every access
and never expose references to graph memory. These wrappers add no new
unsafe behavior.
"""

from __future__ import annotations

import numpy as np

from tradingflow._native import _ArrayView, _SeriesView


class ArrayView:
    """View of a Rust `Array<T>` node in the computation graph.

    All reads copy data out; all writes copy data in. No reference to
    Rust-owned memory is ever exposed to Python.

    Instances are created by the runtime during operator registration —
    users do not construct these directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _ArrayView) -> None:
        self._inner = inner

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

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of the array."""
        return self._inner.shape

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype of the array."""
        return self._inner.dtype

    def __repr__(self) -> str:
        return f"ArrayView(shape={self.shape}, dtype={self.dtype})"


class SeriesView:
    """View of a Rust `Series<T>` node in the computation graph.

    All reads copy data out. Series buffers can reallocate during graph
    execution; copies prevent dangling.

    Instances are created by the runtime during operator registration —
    users do not construct these directly.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: _SeriesView) -> None:
        self._inner = inner

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
