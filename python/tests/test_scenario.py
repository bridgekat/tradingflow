"""Tests for scenario runtime behavior with the new Rust-native backend."""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow import Scenario
from tradingflow.sources import ArraySource
from tradingflow.operators import Add, Multiply, Negate, Record, Filter, Where


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


class TestScenario:
    def test_run_updates_only_affected_downstream_chain(self) -> None:
        """Operators only recompute when at least one of their inputs changed."""
        source_a = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(2)]), values=np.array([1.0, 4.0]), name="a")
        source_b = ArraySource.from_arrays(timestamps=np.array([ts(1)]), values=np.array([2.0]), name="b")
        source_c = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(3)]), values=np.array([10.0, 5.0]), name="c")

        sc = Scenario()
        a = sc.add_source(source_a)
        b = sc.add_source(source_b)
        c = sc.add_source(source_c)

        sum_h = sc.add_operator(Add(a, b))
        scaled_h = sc.add_operator(Multiply(sum_h, c))
        neg_h = sc.add_operator(Negate(scaled_h))

        sum_s = sc.add_operator(Record(sum_h))
        scaled_s = sc.add_operator(Record(scaled_h))
        neg_s = sc.add_operator(Record(neg_h))

        sc.run()

        assert list(sc.series_view(sum_s).timestamps()) == [ts(1), ts(2)]
        assert list(sc.series_view(scaled_s).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(neg_s).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(sum_s).values()) == pytest.approx([3.0, 6.0])
        assert list(sc.series_view(scaled_s).values()) == pytest.approx([30.0, 60.0, 30.0])
        assert list(sc.series_view(neg_s).values()) == pytest.approx([-30.0, -60.0, -30.0])

    def test_same_timestamp_updates_are_coalesced(self) -> None:
        """Two sources at the same timestamp produce a single operator output."""
        source_a = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(1)]), values=np.array([1.0, 2.0]))
        source_b = ArraySource.from_arrays(timestamps=np.array([ts(1)]), values=np.array([3.0]))

        sc = Scenario()
        a = sc.add_source(source_a)
        b = sc.add_source(source_b)
        sum_h = sc.add_operator(Add(a, b))
        sum_s = sc.add_operator(Record(sum_h))

        sc.run()
        assert list(sc.series_view(sum_s).timestamps()) == [ts(1)]
        assert list(sc.series_view(sum_s).values()) == pytest.approx([5.0])

    def test_run_rejects_decreasing_source_timestamps(self) -> None:
        """A source with decreasing timestamps raises at construction time."""
        with pytest.raises(ValueError, match="non-decreasing"):
            ArraySource.from_arrays(timestamps=np.array([ts(2), ts(1)]), values=np.array([1.0, 2.0]))

    def test_output_timestamps_strictly_increasing(self) -> None:
        """Operator output timestamps are strictly increasing."""
        rng = np.random.default_rng(42)
        sources = []
        for i in range(4):
            n_events = int(rng.integers(5, 15))
            raw_ts = np.sort(rng.choice(200, size=n_events, replace=False)).astype("datetime64[ns]")
            values = rng.uniform(0.0, 10.0, size=n_events).astype(np.float64)
            sources.append(ArraySource.from_arrays(timestamps=raw_ts, values=values, name=f"src_{i}"))

        sc = Scenario()
        obs_list = [sc.add_source(src) for src in sources]
        result_h = sc.add_operator(Add(obs_list[0], obs_list[1]))
        for o in obs_list[2:]:
            result_h = sc.add_operator(Add(result_h, o))

        result_s = sc.add_operator(Record(result_h))
        sc.run()

        ts_arr = sc.series_view(result_s).timestamps()
        assert len(ts_arr) > 0
        for i in range(1, len(ts_arr)):
            assert ts_arr[i] > ts_arr[i - 1]

    def test_two_random_sources_add_matches_simulation(self) -> None:
        """`Add(A, B)` matches a pure-Python reference simulation."""
        rng = np.random.default_rng(99)
        raw_ts_a = np.sort(rng.choice(50, size=12, replace=False))
        raw_ts_b = np.sort(rng.choice(50, size=10, replace=False))
        vals_a = rng.uniform(0.0, 5.0, size=len(raw_ts_a))
        vals_b = rng.uniform(0.0, 5.0, size=len(raw_ts_b))

        events_a = dict(zip(raw_ts_a.tolist(), vals_a.tolist()))
        events_b = dict(zip(raw_ts_b.tolist(), vals_b.tolist()))
        all_ts_raw = sorted(set(raw_ts_a.tolist()) | set(raw_ts_b.tolist()))
        exp_ts, exp_vals = [], []
        last_a, last_b = 0.0, 0.0
        for t in all_ts_raw:
            if t in events_a: last_a = events_a[t]
            if t in events_b: last_b = events_b[t]
            exp_ts.append(np.datetime64(t, "ns"))
            exp_vals.append(last_a + last_b)

        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(timestamps=raw_ts_a.astype("datetime64[ns]"), values=vals_a.astype(np.float64), name="a"))
        b = sc.add_source(ArraySource.from_arrays(timestamps=raw_ts_b.astype("datetime64[ns]"), values=vals_b.astype(np.float64), name="b"))
        result_h = sc.add_operator(Add(a, b))
        result_s = sc.add_operator(Record(result_h))
        sc.run()

        assert list(sc.series_view(result_s).timestamps()) == exp_ts
        np.testing.assert_array_almost_equal(list(sc.series_view(result_s).values()), exp_vals)

    def test_interleaved_sources_trigger_output_at_every_timestamp(self) -> None:
        """Every timestamp fires an operator output."""
        ts_a = np.array([1, 3, 5, 7, 9], dtype="datetime64[ns]")
        ts_b = np.array([2, 4, 6, 8, 10], dtype="datetime64[ns]")
        vals_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vals_b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        sc = Scenario()
        a = sc.add_source(ArraySource.from_arrays(timestamps=ts_a, values=vals_a, name="a"))
        b = sc.add_source(ArraySource.from_arrays(timestamps=ts_b, values=vals_b, name="b"))
        result_h = sc.add_operator(Add(a, b))
        result_s = sc.add_operator(Record(result_h))
        sc.run()

        expected_ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="datetime64[ns]")
        expected_vals = [1.0, 11.0, 12.0, 22.0, 23.0, 33.0, 34.0, 44.0, 45.0, 55.0]
        assert list(sc.series_view(result_s).timestamps()) == list(expected_ts)
        np.testing.assert_array_almost_equal(list(sc.series_view(result_s).values()), expected_vals)

    def test_same_source_added_twice_creates_independent_nodes(self) -> None:
        """The same source registered twice creates independent graph nodes."""
        source = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(2), ts(3)]), values=np.array([10.0, 20.0, 30.0]))
        sc = Scenario()
        o1 = sc.add_source(source)
        o2 = sc.add_source(source)
        assert o1 is not o2

        result_h = sc.add_operator(Add(o1, o2))
        s1 = sc.add_operator(Record(o1))
        s2 = sc.add_operator(Record(o2))
        result_s = sc.add_operator(Record(result_h))

        sc.run()

        assert list(sc.series_view(s1).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(s2).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(s1).values()) == pytest.approx([10.0, 20.0, 30.0])
        assert list(sc.series_view(s2).values()) == pytest.approx([10.0, 20.0, 30.0])
        assert list(sc.series_view(result_s).values()) == pytest.approx([20.0, 40.0, 60.0])

    def test_scenario_run_is_repeatable(self) -> None:
        """Running the same scenario structure twice yields identical results."""
        rng = np.random.default_rng(7)
        sources = []
        for i in range(3):
            n = int(rng.integers(8, 15))
            raw_ts = np.sort(rng.choice(100, size=n, replace=False)).astype("datetime64[ns]")
            vals = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
            sources.append(ArraySource.from_arrays(timestamps=raw_ts, values=vals, name=f"s{i}"))

        def build_and_run():
            sc = Scenario()
            obs_list = [sc.add_source(src) for src in sources]
            partial = sc.add_operator(Add(obs_list[0], obs_list[1]))
            result_h = sc.add_operator(Add(partial, obs_list[2]))
            result_s = sc.add_operator(Record(result_h))
            sc.run()
            return list(sc.series_view(result_s).timestamps()), list(sc.series_view(result_s).values())

        ts1, vals1 = build_and_run()
        ts2, vals2 = build_and_run()
        assert ts1 == ts2
        np.testing.assert_array_equal(vals1, vals2)

    def test_filter_stops_downstream_propagation(self) -> None:
        """Filter returning False halts downstream operators."""
        source = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(2), ts(3)]), values=np.array([-1.0, 2.0, -3.0]))
        sc = Scenario()
        s = sc.add_source(source)
        filtered = sc.add_operator(Filter(s, lambda v: float(v.flat[0]) <= 0))
        downstream = sc.add_operator(Negate(filtered))
        fs = sc.add_operator(Record(filtered))
        ds = sc.add_operator(Record(downstream))
        sc.run()

        assert list(sc.series_view(fs).timestamps()) == [ts(1), ts(3)]
        assert list(sc.series_view(fs).values()) == pytest.approx([-1.0, -3.0])
        assert list(sc.series_view(ds).timestamps()) == [ts(1), ts(3)]
        assert list(sc.series_view(ds).values()) == pytest.approx([1.0, 3.0])

    def test_where_replaces_elements(self) -> None:
        """Where replaces failing elements with fill value."""
        source = ArraySource.from_arrays(timestamps=np.array([ts(1)]), values=np.array([[1.0, 5.0, 2.0]]))
        sc = Scenario()
        s = sc.add_source(source)
        w = sc.add_operator(Where(s, lambda x: x > 3.0, fill=0.0))
        ws = sc.add_operator(Record(w))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(ws).values().flatten(), [0.0, 5.0, 0.0])

    def test_python_operator_chained_with_native(self) -> None:
        """A Python Filter feeds into a native negate operator."""
        source = ArraySource.from_arrays(timestamps=np.array([ts(1), ts(2), ts(3), ts(4)]), values=np.array([1.0, 5.0, 2.0, 10.0]))
        sc = Scenario()
        s = sc.add_source(source)
        filtered = sc.add_operator(Filter(s, lambda v: float(v.flat[0]) > 3.0))
        negated = sc.add_operator(Negate(filtered))
        ns = sc.add_operator(Record(negated))
        sc.run()
        assert list(sc.series_view(ns).timestamps()) == [ts(2), ts(4)]
        assert list(sc.series_view(ns).values()) == pytest.approx([-5.0, -10.0])
