"""Tests for internal helpers."""

from __future__ import annotations

import numpy as np

from tradingflow._utils import ensure_contiguous


class TestEnsureContiguous:
    def test_scalar_array_preserves_shape(self) -> None:
        """0-d array must not be promoted to 1-d."""
        a = np.float64(42.0)
        assert a.ndim == 0
        b = ensure_contiguous(a)
        assert b.ndim == 0
        assert b.shape == ()
        assert float(b) == 42.0

    def test_already_contiguous_no_copy(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = ensure_contiguous(a)
        assert b is a

    def test_non_contiguous_becomes_contiguous(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        col = a[:, 0]  # non-contiguous view
        assert not col.flags["C_CONTIGUOUS"]
        b = ensure_contiguous(col)
        assert b.flags["C_CONTIGUOUS"]
        assert b.shape == col.shape
        np.testing.assert_array_equal(b, [1.0, 3.0])

    def test_multidim_preserves_shape(self) -> None:
        a = np.ones((2, 3, 4), dtype=np.float32, order="F")
        assert not a.flags["C_CONTIGUOUS"]
        b = ensure_contiguous(a)
        assert b.flags["C_CONTIGUOUS"]
        assert b.shape == (2, 3, 4)
