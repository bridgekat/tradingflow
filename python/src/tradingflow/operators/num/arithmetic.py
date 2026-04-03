"""Arithmetic and math operators."""

from __future__ import annotations

from ...operator import NativeOperator
from ...types import Handle

# -- Binary arithmetic -------------------------------------------------------


class Add(NativeOperator):
    """Element-wise addition: `a + b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="add", inputs=(a, b), shape=a.shape, dtype=a.dtype)


class Subtract(NativeOperator):
    """Element-wise subtraction: `a - b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="subtract", inputs=(a, b), shape=a.shape, dtype=a.dtype)


class Multiply(NativeOperator):
    """Element-wise multiplication: `a * b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="multiply", inputs=(a, b), shape=a.shape, dtype=a.dtype)


class Divide(NativeOperator):
    """Element-wise division: `a / b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="divide", inputs=(a, b), shape=a.shape, dtype=a.dtype)


# -- Unary arithmetic --------------------------------------------------------


class Negate(NativeOperator):
    """Element-wise negation: `-a`."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="negate", inputs=(a,), shape=a.shape, dtype=a.dtype)


# -- Float unary math --------------------------------------------------------


class Log(NativeOperator):
    """Element-wise natural logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="log", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Log2(NativeOperator):
    """Element-wise base-2 logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="log2", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Log10(NativeOperator):
    """Element-wise base-10 logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="log10", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Exp(NativeOperator):
    """Element-wise exponential."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="exp", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Exp2(NativeOperator):
    """Element-wise base-2 exponential."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="exp2", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Sqrt(NativeOperator):
    """Element-wise square root."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="sqrt", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Ceil(NativeOperator):
    """Element-wise ceiling."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="ceil", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Floor(NativeOperator):
    """Element-wise floor."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="floor", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Round(NativeOperator):
    """Element-wise rounding."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="round", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Recip(NativeOperator):
    """Element-wise reciprocal: `1/a`."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="recip", inputs=(a,), shape=a.shape, dtype=a.dtype)


# -- Signed unary math -------------------------------------------------------


class Abs(NativeOperator):
    """Element-wise absolute value."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="abs", inputs=(a,), shape=a.shape, dtype=a.dtype)


class Sign(NativeOperator):
    """Element-wise signum (-1, 0, or +1)."""

    def __init__(self, a: Handle) -> None:
        super().__init__(kind="sign", inputs=(a,), shape=a.shape, dtype=a.dtype)


# -- Float binary math -------------------------------------------------------


class Min(NativeOperator):
    """Element-wise minimum (IEEE 754 NaN semantics)."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="min", inputs=(a, b), shape=a.shape, dtype=a.dtype)


class Max(NativeOperator):
    """Element-wise maximum (IEEE 754 NaN semantics)."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(kind="max", inputs=(a, b), shape=a.shape, dtype=a.dtype)
