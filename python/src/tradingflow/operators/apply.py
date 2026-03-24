"""Element-wise operator factories.

Each returns a [`NativeOperator`][tradingflow.NativeOperator] descriptor.

Arithmetic: `add`, `subtract`, `multiply`, `divide`, `negate`.

Float math: `log`, `log2`, `log10`, `exp`, `exp2`, `sqrt`,
`ceil`, `floor`, `round`, `recip`.

Signed math: `abs`, `sign`.

Parameterized: `pow`, `scale`, `shift`, `clamp`, `nan_to_num`.

Binary math: `min`, `max`.
"""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle

# =============================================================================
# Arithmetic
# =============================================================================


def add(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise addition: `a + b`."""
    return NativeOperator(kind="add", inputs=(a, b), shape=a.shape, dtype=a.dtype)


def subtract(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise subtraction: `a - b`."""
    return NativeOperator(kind="subtract", inputs=(a, b), shape=a.shape, dtype=a.dtype)


def multiply(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise multiplication: `a * b`."""
    return NativeOperator(kind="multiply", inputs=(a, b), shape=a.shape, dtype=a.dtype)


def divide(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise division: `a / b`."""
    return NativeOperator(kind="divide", inputs=(a, b), shape=a.shape, dtype=a.dtype)


def negate(a: Handle) -> NativeOperator:
    """Element-wise negation: `-a`."""
    return NativeOperator(kind="negate", inputs=(a,), shape=a.shape, dtype=a.dtype)


# =============================================================================
# Float unary math
# =============================================================================


def log(a: Handle) -> NativeOperator:
    """Element-wise natural logarithm."""
    return NativeOperator(kind="log", inputs=(a,), shape=a.shape, dtype=a.dtype)


def log2(a: Handle) -> NativeOperator:
    """Element-wise base-2 logarithm."""
    return NativeOperator(kind="log2", inputs=(a,), shape=a.shape, dtype=a.dtype)


def log10(a: Handle) -> NativeOperator:
    """Element-wise base-10 logarithm."""
    return NativeOperator(kind="log10", inputs=(a,), shape=a.shape, dtype=a.dtype)


def exp(a: Handle) -> NativeOperator:
    """Element-wise exponential."""
    return NativeOperator(kind="exp", inputs=(a,), shape=a.shape, dtype=a.dtype)


def exp2(a: Handle) -> NativeOperator:
    """Element-wise base-2 exponential."""
    return NativeOperator(kind="exp2", inputs=(a,), shape=a.shape, dtype=a.dtype)


def sqrt(a: Handle) -> NativeOperator:
    """Element-wise square root."""
    return NativeOperator(kind="sqrt", inputs=(a,), shape=a.shape, dtype=a.dtype)


def ceil(a: Handle) -> NativeOperator:
    """Element-wise ceiling."""
    return NativeOperator(kind="ceil", inputs=(a,), shape=a.shape, dtype=a.dtype)


def floor(a: Handle) -> NativeOperator:
    """Element-wise floor."""
    return NativeOperator(kind="floor", inputs=(a,), shape=a.shape, dtype=a.dtype)


def round(a: Handle) -> NativeOperator:
    """Element-wise rounding."""
    return NativeOperator(kind="round", inputs=(a,), shape=a.shape, dtype=a.dtype)


def recip(a: Handle) -> NativeOperator:
    """Element-wise reciprocal: `1/a`."""
    return NativeOperator(kind="recip", inputs=(a,), shape=a.shape, dtype=a.dtype)


# =============================================================================
# Signed unary math
# =============================================================================


def abs(a: Handle) -> NativeOperator:
    """Element-wise absolute value."""
    return NativeOperator(kind="abs", inputs=(a,), shape=a.shape, dtype=a.dtype)


def sign(a: Handle) -> NativeOperator:
    """Element-wise signum (-1, 0, or +1)."""
    return NativeOperator(kind="sign", inputs=(a,), shape=a.shape, dtype=a.dtype)


# =============================================================================
# Parameterized unary
# =============================================================================


def pow(a: Handle, n: float) -> NativeOperator:
    """Element-wise power: `a ** n`."""
    return NativeOperator(
        kind="pow", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"n": n}
    )


def scale(a: Handle, c: float) -> NativeOperator:
    """Element-wise scale: `a * c`."""
    return NativeOperator(
        kind="scale", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"c": c}
    )


def shift(a: Handle, c: float) -> NativeOperator:
    """Element-wise shift: `a + c`."""
    return NativeOperator(
        kind="shift", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"c": c}
    )


def clamp(a: Handle, lo: float, hi: float) -> NativeOperator:
    """Element-wise clamp to `[lo, hi]`."""
    return NativeOperator(
        kind="clamp", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"lo": lo, "hi": hi}
    )


def nan_to_num(a: Handle, val: float) -> NativeOperator:
    """Replace NaN with `val`."""
    return NativeOperator(
        kind="nan_to_num", inputs=(a,), shape=a.shape, dtype=a.dtype, params={"val": val}
    )


# =============================================================================
# Float binary math
# =============================================================================


def min(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise minimum (IEEE 754 NaN semantics)."""
    return NativeOperator(kind="min", inputs=(a, b), shape=a.shape, dtype=a.dtype)


def max(a: Handle, b: Handle) -> NativeOperator:
    """Element-wise maximum (IEEE 754 NaN semantics)."""
    return NativeOperator(kind="max", inputs=(a, b), shape=a.shape, dtype=a.dtype)
