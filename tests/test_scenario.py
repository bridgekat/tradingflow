"""Tests for event/scenario runtime behavior."""

from __future__ import annotations

import numpy as np
import pytest

from src import Event, Scenario, Series
from src.ops import add, multiply, negate


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


class TestScenario:
    def test_event_dispatch_updates_only_affected_downstream_chain(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = Series((), np.dtype(np.float64))
        c = Series((), np.dtype(np.float64))

        sum_op = add(a, b)
        scaled_op = multiply(sum_op.output, c)
        neg_op = negate(scaled_op.output)

        scenario = Scenario(sources=(a, b, c), operators=(neg_op, scaled_op, sum_op))

        scenario.dispatch(
            Event(
                ts(1),
                updates={
                    a: np.array(1.0, dtype=np.float64),
                    b: np.array(2.0, dtype=np.float64),
                    c: np.array(10.0, dtype=np.float64),
                },
            )
        )
        assert list(sum_op.output.index) == [ts(1)]
        assert list(scaled_op.output.index) == [ts(1)]
        assert list(neg_op.output.index) == [ts(1)]
        assert list(sum_op.output.values) == pytest.approx([3.0])
        assert list(scaled_op.output.values) == pytest.approx([30.0])
        assert list(neg_op.output.values) == pytest.approx([-30.0])

        scenario.dispatch(
            Event(
                ts(2),
                updates={
                    a: np.array(4.0, dtype=np.float64),
                },
            )
        )
        assert list(sum_op.output.index) == [ts(1), ts(2)]
        assert list(scaled_op.output.index) == [ts(1), ts(2)]
        assert list(neg_op.output.index) == [ts(1), ts(2)]
        assert list(sum_op.output.values) == pytest.approx([3.0, 6.0])
        assert list(scaled_op.output.values) == pytest.approx([30.0, 60.0])
        assert list(neg_op.output.values) == pytest.approx([-30.0, -60.0])

        scenario.dispatch(
            Event(
                ts(3),
                updates={
                    c: np.array(5.0, dtype=np.float64),
                },
            )
        )
        assert list(sum_op.output.index) == [ts(1), ts(2)]
        assert list(scaled_op.output.index) == [ts(1), ts(2), ts(3)]
        assert list(neg_op.output.index) == [ts(1), ts(2), ts(3)]
        assert list(sum_op.output.values) == pytest.approx([3.0, 6.0])
        assert list(scaled_op.output.values) == pytest.approx([30.0, 60.0, 30.0])
        assert list(neg_op.output.values) == pytest.approx([-30.0, -60.0, -30.0])

    def test_dispatch_rejects_unregistered_source_in_event(self) -> None:
        source = Series((), np.dtype(np.float64))
        unknown = Series((), np.dtype(np.float64))
        scenario = Scenario(sources=(source,))

        with pytest.raises(ValueError, match="unregistered source"):
            scenario.dispatch(
                Event(
                    ts(1),
                    updates={unknown: np.array(1.0, dtype=np.float64)},
                )
            )

    def test_dispatch_rejects_non_increasing_event_timestamp(self) -> None:
        source = Series((), np.dtype(np.float64))
        scenario = Scenario(sources=(source,))

        scenario.dispatch(Event(ts(1), updates={source: np.array(1.0, dtype=np.float64)}))
        with pytest.raises(ValueError, match="not greater than the last dispatched"):
            scenario.dispatch(Event(ts(1), updates={source: np.array(2.0, dtype=np.float64)}))

    def test_scenario_freezes_after_first_dispatch(self) -> None:
        source = Series((), np.dtype(np.float64))
        scenario = Scenario(sources=(source,))
        scenario.dispatch(Event(ts(1), updates={source: np.array(1.0, dtype=np.float64)}))

        with pytest.raises(RuntimeError, match="frozen"):
            scenario.add_source(Series((), np.dtype(np.float64)))

        with pytest.raises(RuntimeError, match="frozen"):
            scenario.add_operator(negate(source))

    def test_zero_update_event_is_allowed_and_no_operator_output_is_appended(self) -> None:
        a = Series((), np.dtype(np.float64))
        b = Series((), np.dtype(np.float64))
        sum_op = add(a, b)
        scenario = Scenario(sources=(a, b), operators=(sum_op,))

        scenario.dispatch(Event(ts(1), updates={a: np.array(1.0, dtype=np.float64), b: np.array(2.0, dtype=np.float64)}))
        assert list(sum_op.output.index) == [ts(1)]
        assert list(sum_op.output.values) == pytest.approx([3.0])

        scenario.dispatch(Event(ts(2)))
        assert list(sum_op.output.index) == [ts(1)]
        assert list(sum_op.output.values) == pytest.approx([3.0])

        scenario.dispatch(Event(ts(3), updates={a: np.array(3.0, dtype=np.float64)}))
        assert list(sum_op.output.index) == [ts(1), ts(3)]
        assert list(sum_op.output.values) == pytest.approx([3.0, 5.0])
