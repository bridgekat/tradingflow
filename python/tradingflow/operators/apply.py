"""Generic n-ary apply operator and element-wise arithmetic factories.

[`Apply`][tradingflow.operators.Apply] is a closure-based operator for custom
Python computations.

The arithmetic factories ([`add`][tradingflow.operators.add],
[`subtract`][tradingflow.operators.subtract], etc.) return
[`NativeOperator`][tradingflow.NativeOperator] instances backed by opaque
Rust operator handles.
"""

from __future__ import annotations

from typing import Callable, override

import numpy as np

from ..observable import Observable
from ..operator import Operator, NativeOperator

from tradingflow._native import (
    add as _rust_add,
    subtract as _rust_subtract,
    multiply as _rust_multiply,
    divide as _rust_divide,
    negate as _rust_negate,
)


type _AnyShape = tuple[int, ...]
type _Array[Shape: _AnyShape, T: np.generic] = np.ndarray[Shape, np.dtype[T]]
type _ApplyFn[Shape: _AnyShape, InT: np.generic, OutT: np.generic] = Callable[
    [list[_Array[Shape, InT]]], _Array[Shape, OutT] | None
]


class Apply[Shape: _AnyShape, InT: np.generic, OutT: np.generic](
    Operator[tuple[Observable[Shape, InT], ...], Shape, OutT, None]
):
    """Stateless operator that applies a function to the latest values of input observables.

    At each update, the latest value of every input is collected
    into a list and passed to the user-supplied function.
    """

    __slots__ = ("_fn",)

    _fn: _ApplyFn[Shape, InT, OutT]

    def __init__(
        self,
        inputs: tuple[Observable[Shape, InT], ...],
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
        inputs: tuple[Observable[Shape, InT], ...],
        state: None,
    ) -> tuple[_Array[Shape, OutT] | None, None]:
        values: list[_Array[Shape, InT]] = [inp.last for inp in inputs]
        return self._fn(values), None


# ---------------------------------------------------------------------------
# Arithmetic factory functions
# ---------------------------------------------------------------------------


def add[Shape: _AnyShape, T: np.number](
    a: Observable[Shape, T], b: Observable[Shape, T]
) -> NativeOperator:
    """Element-wise addition: `a + b`."""
    d = str(a.dtype)
    return NativeOperator(lambda: _rust_add(d), (a, b), a.shape, a.dtype)


def subtract[Shape: _AnyShape, T: np.number](
    a: Observable[Shape, T], b: Observable[Shape, T]
) -> NativeOperator:
    """Element-wise subtraction: `a - b`."""
    d = str(a.dtype)
    return NativeOperator(lambda: _rust_subtract(d), (a, b), a.shape, a.dtype)


def negate[Shape: _AnyShape, T: np.number](
    a: Observable[Shape, T],
) -> NativeOperator:
    """Element-wise negation: `-a`."""
    d = str(a.dtype)
    return NativeOperator(lambda: _rust_negate(d), (a,), a.shape, a.dtype)


def multiply[Shape: _AnyShape, T: np.number](
    a: Observable[Shape, T], b: Observable[Shape, T]
) -> NativeOperator:
    """Element-wise multiplication: `a * b`."""
    d = str(a.dtype)
    return NativeOperator(lambda: _rust_multiply(d), (a, b), a.shape, a.dtype)


def divide[Shape: _AnyShape, T: np.floating](
    a: Observable[Shape, T], b: Observable[Shape, T]
) -> NativeOperator:
    """Element-wise division: `a / b` with floating point inputs."""
    d = str(a.dtype)
    return NativeOperator(lambda: _rust_divide(d), (a, b), a.shape, a.dtype)


def map[Shape: _AnyShape, T: np.generic](
    a: Observable[Shape, T],
    fn: Callable[[_Array[Shape, T]], _Array[Shape, T]],
) -> Apply[Shape, T, T]:
    """Unary element-wise transform: `fn(a)`.

    Convenience wrapper for [`Apply`][tradingflow.operators.Apply] with a single input.
    The output has the same shape and dtype as the input.

    Examples
    --------
    ::

        log_obs = scenario.add_operator(map(positive_obs, np.log))
    """
    return Apply((a,), a.shape, a.dtype, lambda args: fn(args[0]))
