"""Generic n-ary apply operator and element-wise arithmetic helper factories.

Classes
-------
Apply[Shape, InT, OutT]
    Stateless operator that applies a user-supplied function to the latest
    values of one or more input series.  Skips output when any input is
    empty.

Factory functions
-----------------
add, subtract, negate, multiply, divide
    Element-wise arithmetic operators that return :class:`Apply` instances.
multiple
    Backward-compatible alias for :func:`divide`.
"""

from __future__ import annotations

from typing import Callable, cast, override

import numpy as np

from ..operator import Operator
from ..series import Series


type _AnyShape = tuple[int, ...]
type _Array[Shape: _AnyShape, T: np.generic] = np.ndarray[Shape, np.dtype[T]]
type _ApplyFn[Shape: _AnyShape, InT: np.generic, OutT: np.generic] = Callable[
    [list[_Array[Shape, InT]]], _Array[Shape, OutT] | None
]


class Apply[Shape: _AnyShape, InT: np.generic, OutT: np.generic](
    Operator[tuple[Series[Shape, InT], ...], Shape, OutT, None]
):
    """Stateless operator that applies a function to the latest values of input series.

    At each :meth:`update`, the latest value of every input is collected
    into a list and passed to the user-supplied function.  If any input
    series is empty the output is skipped (``None``).
    """

    __slots__ = ("_fn",)

    _fn: _ApplyFn[Shape, InT, OutT]

    def __init__(
        self,
        inputs: tuple[Series[Shape, InT], ...],
        shape: Shape,
        dtype: type[OutT] | np.dtype[OutT],
        fn: _ApplyFn[Shape, InT, OutT],
    ) -> None:
        super().__init__(inputs, shape, dtype, None)
        self._fn = fn

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[Shape, InT], ...],
        state: None,
    ) -> _Array[Shape, OutT] | None:
        if not all(inputs):
            return None
        values: list[_Array[Shape, InT]] = [series.values[-1] for series in inputs]
        return self._fn(values)


def _add[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return cast(_Array[Shape, T], x + y)


def _subtract[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return cast(_Array[Shape, T], x - y)


def _negate[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    (x,) = args
    return -x


def _multiply[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return cast(_Array[Shape, T], x * y)


def _divide[Shape: _AnyShape, T: np.floating](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return cast(_Array[Shape, T], x / y)


def add[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise addition: ``a + b``."""
    return Apply((a, b), a.shape, a.dtype, _add)


def subtract[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise subtraction: ``a - b``."""
    return Apply((a, b), a.shape, a.dtype, _subtract)


def negate[Shape: _AnyShape, T: np.number](a: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise negation: ``-a``."""
    return Apply((a,), a.shape, a.dtype, _negate)


def multiply[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise multiplication: ``a * b``."""
    return Apply((a, b), a.shape, a.dtype, _multiply)


def divide[Shape: _AnyShape, T: np.floating](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise division: ``a / b`` with floating point inputs."""
    return Apply((a, b), a.shape, a.dtype, _divide)


def multiple[Shape: _AnyShape, T: np.floating](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Backward-compatible alias for :func:`divide`."""
    return divide(a, b)
