"""Tests for source/scenario runtime behavior."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tradingflow import ArrayBundleSource, AsyncCallableSource, Operator, Scenario
from tradingflow.ops import add, multiply, negate


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


class TestScenario:
    def test_run_updates_only_affected_downstream_chain(self) -> None:
        """Operators only recompute when at least one of their inputs changed.

        Graph: a, b, c → add(a, b) → multiply(sum, c) → negate(scaled)

        * ``sum``    fires whenever a or b updates (t=1, t=2).
        * ``scaled`` fires whenever sum or c updates (t=1, t=2, t=3).
        * ``neg``    mirrors ``scaled`` exactly.

        The key invariant under test is that operators with no new input at a
        given timestamp are skipped, and that carry-forward of the last known
        value is used when only one of two inputs is updated.
        """
        source_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([1.0, 4.0]),
            name="a",
        )
        source_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([2.0]),
            name="b",
        )
        source_c = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(3)]),
            values=np.array([10.0, 5.0]),
            name="c",
        )

        scenario = Scenario()
        a = scenario.add_source(source_a)
        b = scenario.add_source(source_b)
        c = scenario.add_source(source_c)

        sum_series = scenario.add_operator(add(a, b))
        scaled_series = scenario.add_operator(multiply(sum_series, c))
        neg_series = scenario.add_operator(negate(scaled_series))
        asyncio.run(scenario.run())

        assert list(sum_series.index) == [ts(1), ts(2)]
        assert list(scaled_series.index) == [ts(1), ts(2), ts(3)]
        assert list(neg_series.index) == [ts(1), ts(2), ts(3)]
        assert list(sum_series.values) == pytest.approx([3.0, 6.0])
        assert list(scaled_series.values) == pytest.approx([30.0, 60.0, 30.0])
        assert list(neg_series.values) == pytest.approx([-30.0, -60.0, -30.0])

    def test_same_timestamp_updates_are_coalesced(self) -> None:
        """Two sources emitting at the same timestamp produce a single operator output.

        The POCQ accumulates all events sharing a timestamp before flushing, so
        ``add(a, b)`` sees the updated values of both ``a`` and ``b`` in a single
        ``compute`` call rather than computing twice and appending a duplicate entry.
        """
        source_a = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(1)]),
            values=np.array([1.0, 2.0]),
        )
        source_b = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1)]),
            values=np.array([3.0]),
        )

        scenario = Scenario()
        a = scenario.add_source(source_a)
        b = scenario.add_source(source_b)
        sum_series = scenario.add_operator(add(a, b))

        asyncio.run(scenario.run())
        assert list(sum_series.index) == [ts(1)]
        assert list(sum_series.values) == pytest.approx([5.0])

    def test_run_rejects_decreasing_source_timestamps(self) -> None:
        """A source that emits a timestamp smaller than one it already emitted raises.

        Historical iterators must produce non-decreasing timestamps.  The
        runtime validates this on ingest and raises ``ValueError`` immediately,
        which causes the scenario to cancel all pending tasks and re-raise.
        """
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(2), ts(1)]),
            values=np.array([1.0, 2.0]),
        )
        scenario = Scenario()
        scenario.add_source(source)

        with pytest.raises(ValueError, match="less than last committed timestamp"):
            asyncio.run(scenario.run())

    def test_run_fail_fast_when_source_raises(self) -> None:
        """An exception raised inside a live source iterator propagates out of run().

        The scenario must not swallow the error or deadlock; it should cancel all
        other in-flight tasks and re-raise the original exception.
        """
        payload_source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([1.0, 2.0]),
        )

        async def failing_stream():
            yield 10.0
            raise RuntimeError("boom")

        realtime_source = AsyncCallableSource((), np.float64, failing_stream)
        scenario = Scenario()
        scenario.add_source(payload_source)
        scenario.add_source(realtime_source)

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(scenario.run())

    def test_output_timestamps_strictly_increasing(self) -> None:
        """Operator output timestamps are strictly increasing regardless of source interleaving.

        Four random historical sources are chained with ``add`` operators.  After
        ``run()``, every consecutive pair of timestamps in the final output series
        must satisfy ``t[i] < t[i+1]``.  This guards against any POCQ flush
        ordering bug that could allow an earlier timestamp to be appended after a
        later one.
        """
        rng = np.random.default_rng(42)
        sources = []
        for i in range(4):
            n_events = int(rng.integers(5, 15))
            raw_ts = np.sort(rng.choice(200, size=n_events, replace=False)).astype("datetime64[ns]")
            values = rng.uniform(0.0, 10.0, size=n_events).astype(np.float64)
            sources.append(ArrayBundleSource.from_arrays(timestamps=raw_ts, values=values, name=f"src_{i}"))

        scenario = Scenario()
        series = [scenario.add_source(src) for src in sources]
        result = scenario.add_operator(add(series[0], series[1]))
        for s in series[2:]:
            result = scenario.add_operator(add(result, s))

        asyncio.run(scenario.run())

        assert len(result.index) > 0
        for i in range(1, len(result.index)):
            assert result.index[i] > result.index[i - 1], (
                f"Non-increasing timestamps at positions {i - 1} and {i}: "
                f"{result.index[i - 1]} >= {result.index[i]}"
            )

    def test_two_random_sources_add_matches_simulation(self) -> None:
        """``add(A, B)`` output matches a pure-Python reference simulation of the POCQ.

        The reference model iterates the union of both sources' timestamps in order,
        tracking the last known value for each source and emitting ``last_a + last_b``
        once both have been seen at least once.  The scenario result must agree
        exactly on both timestamps and values.
        """
        rng = np.random.default_rng(99)

        raw_ts_a = np.sort(rng.choice(50, size=12, replace=False))
        raw_ts_b = np.sort(rng.choice(50, size=10, replace=False))
        vals_a = rng.uniform(0.0, 5.0, size=len(raw_ts_a))
        vals_b = rng.uniform(0.0, 5.0, size=len(raw_ts_b))

        # Reference simulation: at each timestamp in the union, track last known values.
        events_a = dict(zip(raw_ts_a.tolist(), vals_a.tolist()))
        events_b = dict(zip(raw_ts_b.tolist(), vals_b.tolist()))
        all_ts_raw = sorted(set(raw_ts_a.tolist()) | set(raw_ts_b.tolist()))
        exp_ts: list[np.datetime64] = []
        exp_vals: list[float] = []
        last_a = last_b = None
        for t in all_ts_raw:
            if t in events_a:
                last_a = events_a[t]
            if t in events_b:
                last_b = events_b[t]
            if last_a is not None and last_b is not None:
                exp_ts.append(np.datetime64(t, "ns"))
                exp_vals.append(last_a + last_b)

        src_a = ArrayBundleSource.from_arrays(
            timestamps=raw_ts_a.astype("datetime64[ns]"),
            values=vals_a.astype(np.float64),
            name="a",
        )
        src_b = ArrayBundleSource.from_arrays(
            timestamps=raw_ts_b.astype("datetime64[ns]"),
            values=vals_b.astype(np.float64),
            name="b",
        )

        scenario = Scenario()
        a = scenario.add_source(src_a)
        b = scenario.add_source(src_b)
        result = scenario.add_operator(add(a, b))
        asyncio.run(scenario.run())

        assert list(result.index) == exp_ts
        assert list(result.values) == pytest.approx(exp_vals)

    def test_interleaved_sources_trigger_output_at_every_timestamp(self) -> None:
        """Every timestamp fires an operator output once both inputs have been seen.

        Source A emits at odd nanoseconds [1, 3, 5, 7, 9] and source B at even
        nanoseconds [2, 4, 6, 8, 10].  ``add(A, B)`` cannot fire at t=1 because B
        has no value yet, but from t=2 onward every new event from either source
        triggers a recompute using the last known value of the other input.  The
        expected output therefore spans t=2..10 with carry-forward semantics verified
        at each step.
        """
        # A fires at odd nanoseconds, B fires at even nanoseconds.
        ts_a = np.array([1, 3, 5, 7, 9], dtype="datetime64[ns]")
        ts_b = np.array([2, 4, 6, 8, 10], dtype="datetime64[ns]")
        vals_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vals_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        scenario = Scenario()
        a = scenario.add_source(ArrayBundleSource.from_arrays(timestamps=ts_a, values=vals_a, name="a"))
        b = scenario.add_source(ArrayBundleSource.from_arrays(timestamps=ts_b, values=vals_b, name="b"))
        result = scenario.add_operator(add(a, b))
        asyncio.run(scenario.run())

        # t=1: only A known → no output yet.
        # From t=2 onward both are known, so every timestamp fires.
        expected_ts = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="datetime64[ns]")
        expected_vals = [
            1.0 + 10.0,  # t=2:  a_last=1, b=10
            2.0 + 10.0,  # t=3:  a=2,      b_last=10
            2.0 + 20.0,  # t=4:  a_last=2, b=20
            3.0 + 20.0,  # t=5:  a=3,      b_last=20
            3.0 + 30.0,  # t=6:  a_last=3, b=30
            4.0 + 30.0,  # t=7:  a=4,      b_last=30
            4.0 + 40.0,  # t=8:  a_last=4, b=40
            5.0 + 40.0,  # t=9:  a=5,      b_last=40
            5.0 + 50.0,  # t=10: a_last=5, b=50
        ]

        assert list(result.index) == list(expected_ts)
        assert list(result.values) == pytest.approx(expected_vals)

    def test_same_source_added_twice_creates_independent_nodes(self) -> None:
        """The same source object registered twice creates two fully independent graph nodes.

        Each call to ``add_source`` allocates a fresh output series backed by its
        own iterator pair and per-run state.  Both series must receive the complete
        data from the underlying source independently, and a downstream operator
        consuming both must see them as distinct inputs.
        """
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([10.0, 20.0, 30.0]),
        )
        scenario = Scenario()
        s1 = scenario.add_source(source)
        s2 = scenario.add_source(source)

        assert s1 is not s2

        result = scenario.add_operator(add(s1, s2))
        asyncio.run(scenario.run())

        # Both nodes should receive the full data independently.
        assert list(s1.index) == [ts(1), ts(2), ts(3)]
        assert list(s2.index) == [ts(1), ts(2), ts(3)]
        assert list(s1.values) == pytest.approx([10.0, 20.0, 30.0])
        assert list(s2.values) == pytest.approx([10.0, 20.0, 30.0])
        # add(s1, s2) = 2 * source value at every timestamp.
        assert list(result.index) == [ts(1), ts(2), ts(3)]
        assert list(result.values) == pytest.approx([20.0, 40.0, 60.0])

    def test_same_operator_added_twice_creates_independent_nodes(self) -> None:
        """The same operator object registered twice creates two fully independent output nodes.

        Each call to ``add_operator`` allocates a fresh output series and a separate
        computation state initialised via ``init_state()``.  Both output series must
        be populated identically and independently; neither should be empty or missing
        entries because the other was registered.
        """
        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2)]),
            values=np.array([3.0, 7.0]),
        )
        scenario = Scenario()
        s = scenario.add_source(source)
        neg_op = negate(s)
        r1 = scenario.add_operator(neg_op)
        r2 = scenario.add_operator(neg_op)

        assert r1 is not r2

        asyncio.run(scenario.run())

        assert list(r1.index) == [ts(1), ts(2)]
        assert list(r2.index) == [ts(1), ts(2)]
        assert list(r1.values) == pytest.approx([-3.0, -7.0])
        assert list(r2.values) == pytest.approx([-3.0, -7.0])

    def test_scenario_run_is_repeatable(self) -> None:
        """Building and running the same scenario twice yields bit-identical results.

        ``Scenario.run()`` creates a fresh ``_ScenarioState`` per invocation, so
        source iterators are reset and operator states are re-initialised each time.
        Two independent scenario builds from the same source objects must produce
        the same output timestamps and values, confirming that no mutable state
        leaks between runs.
        """
        rng = np.random.default_rng(7)
        sources = []
        for i in range(3):
            n = int(rng.integers(8, 15))
            raw_ts = np.sort(rng.choice(100, size=n, replace=False)).astype("datetime64[ns]")
            vals = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
            sources.append(ArrayBundleSource.from_arrays(timestamps=raw_ts, values=vals, name=f"s{i}"))

        def build_and_run() -> tuple[list, list]:
            scenario = Scenario()
            series = [scenario.add_source(src) for src in sources]
            partial = scenario.add_operator(add(series[0], series[1]))
            result = scenario.add_operator(add(partial, series[2]))
            asyncio.run(scenario.run())
            return list(result.index), list(result.values)

        ts1, vals1 = build_and_run()
        ts2, vals2 = build_and_run()

        assert ts1 == ts2
        assert vals1 == pytest.approx(vals2)

    def test_operator_returning_none_stops_downstream_propagation(self) -> None:
        """An operator returning ``None`` suppresses its output and halts downstream propagation.

        When ``compute`` returns ``(None, state)``, the scenario must not append
        any entry to the operator's output series, and must not notify exclusively-
        downstream operators that an update occurred at that timestamp.

        The test uses a custom ``_FilterPositive`` operator that passes through
        non-positive values and returns ``None`` for positive ones, then chains it
        with ``negate``.  At t=2 the source value is +2.0, so ``_FilterPositive``
        returns ``None`` and ``negate`` must remain silent; at t=1 and t=3 both
        operators fire normally.
        """

        class _FilterPositive(Operator):
            """Passes through values <= 0; returns None for positive values."""

            def __init__(self, inp: object) -> None:
                super().__init__((inp,), (), np.float64)  # type: ignore[arg-type]

            def init_state(self) -> None:
                return None

            def compute(self, timestamp: np.datetime64, inputs: object, state: None) -> tuple:
                val = float(inputs[0].values[-1])  # type: ignore[index]
                return (None, None) if val > 0 else (np.float64(val), None)

        source = ArrayBundleSource.from_arrays(
            timestamps=np.array([ts(1), ts(2), ts(3)]),
            values=np.array([-1.0, 2.0, -3.0]),
        )
        scenario = Scenario()
        s = scenario.add_source(source)
        filtered = scenario.add_operator(_FilterPositive(s))
        downstream = scenario.add_operator(negate(filtered))
        asyncio.run(scenario.run())

        # t=1: -1.0 passes through → filtered emits, downstream emits
        # t=2:  2.0 is positive  → filtered returns None → no entry, downstream silent
        # t=3: -3.0 passes through → filtered emits, downstream emits
        assert list(filtered.index) == [ts(1), ts(3)]
        assert list(filtered.values) == pytest.approx([-1.0, -3.0])
        assert list(downstream.index) == [ts(1), ts(3)]
        assert list(downstream.values) == pytest.approx([1.0, 3.0])
