"""Tests for source/scenario runtime behavior."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src import ArrayBundleSource, AsyncCallableSource, Scenario, Series
from src.ops import add, multiply, negate


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


class TestScenario:
    def test_run_updates_only_affected_downstream_chain(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = Series((), np.dtype(np.float64))
        c = Series((), np.dtype(np.float64))

        sum_op = add(a, b)
        scaled_op = multiply(sum_op.output, c)
        neg_op = negate(scaled_op.output)

        source_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([1.0, 4.0]),
            series=a,
            name="a",
        )
        source_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([2.0]),
            series=b,
            name="b",
        )
        source_c = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(3)]),
            values=np.array([10.0, 5.0]),
            series=c,
            name="c",
        )

        scenario = Scenario(sources=(source_a, source_b, source_c), operators=(neg_op, scaled_op, sum_op))
        asyncio.run(scenario.run())

        assert list(sum_op.output.index) == [ts(1), ts(2)]
        assert list(scaled_op.output.index) == [ts(1), ts(2), ts(3)]
        assert list(neg_op.output.index) == [ts(1), ts(2), ts(3)]
        assert list(sum_op.output.values) == pytest.approx([3.0, 6.0])
        assert list(scaled_op.output.values) == pytest.approx([30.0, 60.0, 30.0])
        assert list(neg_op.output.values) == pytest.approx([-30.0, -60.0, -30.0])

    def test_same_timestamp_updates_are_coalesced(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = Series((), np.dtype(np.float64))
        sum_op = add(a, b)

        source_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([1.0]),
            series=a,
        )
        source_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([2.0]),
            series=b,
        )
        scenario = Scenario(sources=(source_a, source_b), operators=(sum_op,))

        asyncio.run(scenario.run())
        assert list(sum_op.output.index) == [ts(1)]
        assert list(sum_op.output.values) == pytest.approx([3.0])

    def test_run_rejects_non_increasing_source_timestamps(self) -> None:
        source_series = Series((), np.dtype(np.float64))
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(1)]),
            values=np.array([1.0, 2.0]),
            series=source_series,
        )
        scenario = Scenario(sources=(source,))

        with pytest.raises(ValueError, match="non-increasing timestamp"):
            asyncio.run(scenario.run())

    def test_scenario_freezes_after_first_run(self) -> None:
        source_series = Series((), np.dtype(np.float64))
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([1.0]),
            series=source_series,
        )
        scenario = Scenario(sources=(source,))
        asyncio.run(scenario.run())

        with pytest.raises(RuntimeError, match="frozen"):
            scenario.add_source(
                ArrayBundleSource.from_arrays(
                    timestamps=np.array([ts(2)]),
                    values=np.array([2.0]),
                    series=Series((), np.dtype(np.float64)),
                )
            )

        with pytest.raises(RuntimeError, match="frozen"):
            scenario.add_operator(negate(source_series))

    def test_run_fail_fast_when_source_raises(self) -> None:
        payload_series = Series((), np.dtype(np.float64))
        realtime_series = Series((), np.dtype(np.float64))

        payload_source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([1.0, 2.0]),
            series=payload_series,
        )

        async def failing_stream():
            yield 10.0
            raise RuntimeError("boom")

        realtime_source = AsyncCallableSource(realtime_series, failing_stream)
        scenario = Scenario(sources=(payload_source, realtime_source))

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(scenario.run())
