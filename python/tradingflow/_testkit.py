"""Pure-Python fakes of the Rust host views, for testing ported operators.

`FakeArrayView` / `FakeSeriesView` mirror the API of the Rust `NativeArrayView` /
`NativeSeriesView` so an operator can be exercised (init + compute) with
synthetic NumPy data, validating the Python logic without the Rust engine.

Only *input* views need fakes. Outputs are whatever `compute` returns (an
array, or `None` for no event), so a test reads the return value directly.
Signal leaves need no fake either: a rank-0 pulse slot in the unified `inputs`
tuple is truthiness-tested only (`if signal:`), so tests pass plain Python
bools where the engine would pass a `NativeArrayViewBool`.
"""

from __future__ import annotations

import numpy as np


class FakeArrayView:
    """Mirror of `NativeArrayView` (a read-only `Array<f64>` cell view)."""

    def __init__(self, arr) -> None:
        self._arr = np.asarray(arr, dtype=np.float64)

    def value(self) -> np.ndarray:
        return self._arr.copy()

    def to_numpy(self) -> np.ndarray:
        return self.value()

    def __array__(self, dtype=None, copy=None):
        a = self.value()
        return a if dtype is None else a.astype(dtype)

    @property
    def shape(self):
        return self._arr.shape


class FakeSeriesView:
    """Mirror of `NativeSeriesView` (a `Series<f64>` cell view)."""

    def __init__(self, values, timestamps=None, elem_shape=None) -> None:
        self._v = [np.asarray(x, dtype=np.float64) for x in values]
        self._ts = list(timestamps) if timestamps is not None else list(range(len(self._v)))
        if elem_shape is not None:
            self._shape = tuple(elem_shape)
        else:
            self._shape = tuple(self._v[0].shape) if self._v else ()

    def __len__(self) -> int:
        return len(self._v)

    @property
    def shape(self):
        return self._shape

    def values(self, start: int = 0, end: int | None = None) -> np.ndarray:
        end = len(self._v) if end is None else end
        sub = self._v[start:end]
        if sub:
            return np.stack(sub)
        return np.empty((0, *self._shape), dtype=np.float64)

    def last(self) -> np.ndarray:
        return self._v[-1].copy()

    def at(self, i: int) -> np.ndarray:
        return self._v[i].copy()

    def slice(self, start: int = 0, end: int | None = None) -> np.ndarray:
        end = len(self._ts) if end is None else end
        return np.asarray(self._ts[start:end], dtype=np.int64)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, _ = key.indices(len(self._v))
            return self.values(start, stop)
        return self.at(key)
