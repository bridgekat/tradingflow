"""Arithmetic and math operators."""

from __future__ import annotations

from ... import Handle, NativeOperator, NodeKind

# -- Binary arithmetic -------------------------------------------------------


class Add(NativeOperator):
    """Element-wise addition: `a + b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="add", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Subtract(NativeOperator):
    """Element-wise subtraction: `a - b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="subtract", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Multiply(NativeOperator):
    """Element-wise multiplication: `a * b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="multiply", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Divide(NativeOperator):
    """Element-wise division: `a / b`."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="divide", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


# -- Unary arithmetic --------------------------------------------------------


class Negate(NativeOperator):
    """Element-wise negation: `-a`."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="negate", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


# -- Float unary math --------------------------------------------------------


class Log(NativeOperator):
    """Element-wise natural logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="log", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Log2(NativeOperator):
    """Element-wise base-2 logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="log2", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Log10(NativeOperator):
    """Element-wise base-10 logarithm."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="log10", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Exp(NativeOperator):
    """Element-wise exponential."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="exp", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Exp2(NativeOperator):
    """Element-wise base-2 exponential."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="exp2", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Sqrt(NativeOperator):
    """Element-wise square root."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="sqrt", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Ceil(NativeOperator):
    """Element-wise ceiling."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="ceil", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Floor(NativeOperator):
    """Element-wise floor."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="floor", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Round(NativeOperator):
    """Element-wise rounding."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="round", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Recip(NativeOperator):
    """Element-wise reciprocal: `1/a`."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="recip", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


# -- Signed unary math -------------------------------------------------------


class Abs(NativeOperator):
    """Element-wise absolute value."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="abs", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Sign(NativeOperator):
    """Element-wise signum (-1, 0, or +1)."""

    def __init__(self, a: Handle) -> None:
        super().__init__(native_id="sign", inputs=(a,), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


# -- Float binary math -------------------------------------------------------


class Min(NativeOperator):
    """Element-wise minimum (IEEE 754 NaN semantics)."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="min", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)


class Max(NativeOperator):
    """Element-wise maximum (IEEE 754 NaN semantics)."""

    def __init__(self, a: Handle, b: Handle) -> None:
        super().__init__(native_id="max", inputs=(a, b), kind=NodeKind.ARRAY, dtype=a.dtype, shape=a.shape)
