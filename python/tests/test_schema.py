"""Tests for the pure-Python Schema class."""

from __future__ import annotations

import pytest

from tradingflow import Schema


class TestSchemaBasic:
    def test_len_and_lookup(self):
        s = Schema(["a", "b", "c"])
        assert len(s) == 3
        assert s.index("a") == 0
        assert s.index("c") == 2
        assert s.name(1) == "b"

    def test_contains(self):
        s = Schema(["a", "b", "c"])
        assert s.contains("b")
        assert not s.contains("d")

    def test_empty(self):
        s = Schema([])
        assert len(s) == 0

    def test_repr(self):
        s = Schema(["x", "y"])
        assert repr(s) == "Schema(['x', 'y'])"


class TestSchemaIndices:
    def test_indices(self):
        s = Schema(["x", "y", "z"])
        assert s.indices(["z", "x"]) == [2, 0]

    def test_indices_empty(self):
        s = Schema(["a", "b"])
        assert s.indices([]) == []


class TestSchemaTryIndex:
    def test_present(self):
        s = Schema(["a", "b"])
        assert s.try_index("a") == 0

    def test_absent(self):
        s = Schema(["a", "b"])
        assert s.try_index("missing") is None


class TestSchemaSelect:
    def test_select(self):
        s = Schema(["a", "b", "c", "d"])
        sub = s.select([1, 3])
        assert sub.names == ["b", "d"]
        assert sub.index("d") == 1

    def test_select_empty(self):
        s = Schema(["a", "b"])
        sub = s.select([])
        assert len(sub) == 0


class TestSchemaConcat:
    def test_concat(self):
        s1 = Schema(["a", "b"])
        s2 = Schema(["c", "d"])
        merged = s1.concat(s2)
        assert len(merged) == 4
        assert merged.index("c") == 2

    def test_concat_overlap_raises(self):
        s1 = Schema(["a", "b"])
        s2 = Schema(["b", "c"])
        with pytest.raises(ValueError, match="duplicate name"):
            s1.concat(s2)


class TestSchemaErrors:
    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="duplicate name"):
            Schema(["a", "b", "a"])

    def test_missing_index_raises(self):
        s = Schema(["a", "b"])
        with pytest.raises(KeyError):
            s.index("missing")

    def test_out_of_bounds_name_raises(self):
        s = Schema(["a", "b"])
        with pytest.raises(IndexError):
            s.name(5)
