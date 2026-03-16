"""Generic n-ary apply operator and element-wise arithmetic helper factories."""

from __future__ import annotations

from typing import Callable, override

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

    At each update, the latest value of every input is collected
    into a list and passed to the user-supplied function.  If any input
    series is empty the output is skipped (`None`).
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
        super().__init__(inputs, shape, dtype)
        self._fn = fn

    @override
    def init_state(self) -> None:
        return None

    @override
    def compute(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series[Shape, InT], ...],
        state: None,
    ) -> tuple[_Array[Shape, OutT] | None, None]:
        if not all(inputs):
            return None, None
        values: list[_Array[Shape, InT]] = [series.last for series in inputs]
        return self._fn(values), None


def _add[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return x + y  # type: ignore[return-value]


def _subtract[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return x - y  # type: ignore[return-value]


def _negate[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    (x,) = args
    return -x


def _multiply[Shape: _AnyShape, T: np.number](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return x * y  # type: ignore[return-value]


def _divide[Shape: _AnyShape, T: np.floating](args: list[_Array[Shape, T]]) -> _Array[Shape, T]:
    x, y = args
    return x / y  # type: ignore[return-value]


def add[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise addition: `a + b`."""
    return Apply((a, b), a.shape, a.dtype, _add)


def subtract[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise subtraction: `a - b`."""
    return Apply((a, b), a.shape, a.dtype, _subtract)


def negate[Shape: _AnyShape, T: np.number](a: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise negation: `-a`."""
    return Apply((a,), a.shape, a.dtype, _negate)


def multiply[Shape: _AnyShape, T: np.number](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise multiplication: `a * b`."""
    return Apply((a, b), a.shape, a.dtype, _multiply)


def divide[Shape: _AnyShape, T: np.floating](a: Series[Shape, T], b: Series[Shape, T]) -> Apply[Shape, T, T]:
    """Element-wise division: `a / b` with floating point inputs."""
    return Apply((a, b), a.shape, a.dtype, _divide)


def map[Shape: _AnyShape, T: np.generic](
    a: Series[Shape, T],
    fn: Callable[[_Array[Shape, T]], _Array[Shape, T]],
) -> Apply[Shape, T, T]:
    """Unary element-wise transform: `fn(a)`.

    Convenience wrapper for [`Apply`][tradingflow.operators.Apply] with a single input.
    The output series has the same shape and dtype as the input.

    Examples
    --------
    ::

        log_s = scenario.add_operator(map(positive_s, np.log))
    """
    return Apply((a,), a.shape, a.dtype, lambda args: fn(args[0]))
