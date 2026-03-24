"""Tests for source implementations."""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow import Scenario
from tradingflow.operators import add, record
from tradingflow.sources import ArraySource, IterSource, clock


def ts(i: int) -> np.datetime64:
    return np.datetime64(i, "ns")


class TestIterSource:
    def test_basic(self) -> None:
        """IterSource feeds events from a list of (timestamp, value) pairs."""
        sc = Scenario()
        src = IterSource(
            [(ts(1), 10.0), (ts(2), 20.0), (ts(3), 30.0)],
            shape=(), dtype=np.float64,
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        sc.run()

        assert sc.series_len(s) == 3
        np.testing.assert_array_almost_equal(sc.series_values(s), [10.0, 20.0, 30.0])

    def test_vector_values(self) -> None:
        """IterSource works with vector-valued elements."""
        sc = Scenario()
        src = IterSource(
            [(ts(1), [1.0, 2.0]), (ts(2), [3.0, 4.0])],
            shape=(2,), dtype=np.float64,
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        sc.run()

        assert sc.series_len(s) == 2
        np.testing.assert_array_almost_equal(
            sc.series_values(s).flatten(), [1.0, 2.0, 3.0, 4.0]
        )

    def test_with_operator(self) -> None:
        """IterSource feeds into a native add operator."""
        sc = Scenario()
        a = sc.add_source(IterSource([(ts(1), 10.0), (ts(2), 20.0)], shape=(), dtype=np.float64))
        b = sc.add_source(IterSource([(ts(1), 1.0), (ts(2), 2.0)], shape=(), dtype=np.float64))
        c = sc.add_operator(add(a, b))
        s = sc.add_operator(record(c))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_values(s), [11.0, 22.0])

    def test_repeated_run(self) -> None:
        """IterSource can be used across multiple scenario runs."""
        data = [(ts(1), 5.0), (ts(2), 10.0)]
        for _ in range(2):
            sc = Scenario()
            h = sc.add_source(IterSource(data, shape=(), dtype=np.float64))
            s = sc.add_operator(record(h))
            sc.run()
            np.testing.assert_array_almost_equal(sc.series_values(s), [5.0, 10.0])

    def test_generator_expression(self) -> None:
        """IterSource works with a generator expression (consumed to list at init)."""
        sc = Scenario()
        src = IterSource(
            ((ts(i), float(i * 10)) for i in range(1, 4)),
            shape=(), dtype=np.float64,
        )
        h = sc.add_source(src)
        s = sc.add_operator(record(h))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_values(s), [10.0, 20.0, 30.0])


class TestClockSource:
    def test_clock_registers(self) -> None:
        """Clock source can be registered without errors."""
        sc = Scenario()
        clk = sc.add_source(clock([ts(1), ts(2), ts(3)]))
        assert clk.index >= 0

    def test_clock_with_array_source(self) -> None:
        """Clock source coexists with data sources in the same scenario."""
        sc = Scenario()
        data = sc.add_source(ArraySource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([10.0, 20.0, 30.0]),
        ))
        _clk = sc.add_source(clock([ts(2)]))
        s = sc.add_operator(record(data))
        sc.run()

        assert sc.series_len(s) == 3
        np.testing.assert_array_almost_equal(sc.series_values(s), [10.0, 20.0, 30.0])
