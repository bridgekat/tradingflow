"""Element-wise arithmetic operator factories.

Each returns a [`NativeOperator`][tradingflow.NativeOperator] descriptor.
"""

from __future__ import annotations

from ..operator import NativeOperator
from ..types import Handle


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
