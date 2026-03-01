"""Generic n-ary apply operator and arithmetic helper factories."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..operator import Operator
from ..series import T, Series


class Apply(Operator[None, T]):
    """Applies a function to the latest values of input series."""

    __slots__ = ("_fn",)

    def __init__(
        self,
        inputs: list[Series[Any]],
        fn: Callable[..., Optional[ArrayLike]],
        dtype: np.dtype[T],
        shape: tuple[int, ...] = (),
    ) -> None:
        super().__init__(inputs, None, dtype, shape)
        self._fn = fn

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        if not all(inputs):
            return None
        values = tuple(s.values[-1] for s in inputs)
        return self._fn(*values)


def add(a: Series[T], b: Series[T]) -> Apply[T]:
    """Element-wise addition: ``a + b``."""
    return Apply([a, b], lambda x, y: x + y, a.values.dtype, a.shape)


def subtract(a: Series[T], b: Series[T]) -> Apply[T]:
    """Element-wise subtraction: ``a - b``."""
    return Apply([a, b], lambda x, y: x - y, a.values.dtype, a.shape)


def multiply(a: Series[T], b: Series[T]) -> Apply[T]:
    """Element-wise multiplication: ``a * b``."""
    return Apply([a, b], lambda x, y: x * y, a.values.dtype, a.shape)


def divide(a: Series[Any], b: Series[Any]) -> Apply[np.float64]:
    """Element-wise division: ``a / b`` with float64 output."""
    return Apply([a, b], lambda x, y: x / y, np.dtype(np.float64), a.shape)


def multiple(a: Series[Any], b: Series[Any]) -> Apply[np.float64]:
    """Computes the price multiple (ratio): ``a / b``."""
    return divide(a, b)


def negate(a: Series[T]) -> Apply[T]:
    """Element-wise negation: ``-a``."""
    return Apply([a], lambda x: -x, a.values.dtype, a.shape)
