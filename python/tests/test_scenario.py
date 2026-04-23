"""Tests for scenario runtime behavior."""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow import Scenario
from tradingflow.sources import ArraySource, IterSource
from tradingflow.operators import Record, Filter, Where, Select, Concat
from tradingflow.operators.num import Add, Subtract, Multiply, Divide, Negate


def ts(i: int) -> np.datetime64:
    """Create a nanosecond timestamp from an integer."""
    return np.datetime64(i, "ns")


# =========================================================================
# Native sources + native operators
# =========================================================================


class TestNativeOperators:
    """Basic arithmetic and structural operators with native sources."""

    def test_add_and_record(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(2)], [10.0, 20.0]))
        hb = sc.add_source(ArraySource([ts(1), ts(2)], [1.0, 2.0]))
        hc = sc.add_operator(Add(ha, hb))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).values()) == pytest.approx([11.0, 22.0])

    def test_subtract(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1)], [10.0]))
        hb = sc.add_source(ArraySource([ts(1)], [3.0]))
        hc = sc.add_operator(Subtract(ha, hb))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).values()) == pytest.approx([7.0])

    def test_multiply(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1)], [4.0]))
        hb = sc.add_source(ArraySource([ts(1)], [5.0]))
        hc = sc.add_operator(Multiply(ha, hb))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).values()) == pytest.approx([20.0])

    def test_negate(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(2)], [10.0, -5.0]))
        hc = sc.add_operator(Negate(ha))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).values()) == pytest.approx([-10.0, 5.0])

    def test_chained_operators(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(2)], [2.0, 5.0]))
        hb = sc.add_source(ArraySource([ts(1), ts(2)], [3.0, 10.0]))
        hab = sc.add_operator(Add(ha, hb))
        hout = sc.add_operator(Multiply(hab, ha))
        hs = sc.add_operator(Record(hout))
        sc.run()
        # (2+3)*2=10, (5+10)*5=75
        assert list(sc.series_view(hs).values()) == pytest.approx([10.0, 75.0])

    def test_interleaved_sources(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(3)], [10.0, 30.0]))
        hb = sc.add_source(ArraySource([ts(2), ts(3)], [20.0, 40.0]))
        hc = sc.add_operator(Add(ha, hb))
        hs = sc.add_operator(Record(hc))
        sc.run()
        # ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert list(sc.series_view(hs).values()) == pytest.approx([10.0, 30.0, 70.0])

    def test_strided_add(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0]]))
        hb = sc.add_source(ArraySource([ts(1)], [[10.0, 20.0]]))
        hc = sc.add_operator(Add(ha, hb))
        hs = sc.add_operator(Record(hc))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(hs).values().flatten(), [11.0, 22.0])

    def test_select(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1)], [[10.0, 20.0, 30.0]]))
        hc = sc.add_operator(Select(ha, [0, 2]))
        hs = sc.add_operator(Record(hc))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(hs).values().flatten(), [10.0, 30.0])

    def test_concat(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0]]))
        hb = sc.add_source(ArraySource([ts(1)], [[3.0, 4.0]]))
        hc = sc.add_operator(Concat([ha, hb]))
        hs = sc.add_operator(Record(hc))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(hs).values().flatten(), [1.0, 2.0, 3.0, 4.0])


# =========================================================================
# Python operators
# =========================================================================


class TestPythonOperators:
    """Python-implemented operators registered via the Scenario API."""

    def test_filter_stops_downstream_propagation(self) -> None:
        source = ArraySource([ts(1), ts(2), ts(3)], [-1.0, 2.0, -3.0])
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
        source = ArraySource([ts(1)], [[1.0, 5.0, 2.0]])
        sc = Scenario()
        s = sc.add_source(source)
        w = sc.add_operator(Where(s, lambda x: x > 3.0, fill=0.0))
        ws = sc.add_operator(Record(w))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(ws).values().flatten(), [0.0, 5.0, 0.0])

    def test_python_operator_chained_with_native(self) -> None:
        source = ArraySource([ts(1), ts(2), ts(3), ts(4)], [1.0, 5.0, 2.0, 10.0])
        sc = Scenario()
        s = sc.add_source(source)
        filtered = sc.add_operator(Filter(s, lambda v: float(v.flat[0]) > 3.0))
        negated = sc.add_operator(Negate(filtered))
        ns = sc.add_operator(Record(negated))
        sc.run()
        assert list(sc.series_view(ns).timestamps()) == [ts(2), ts(4)]
        assert list(sc.series_view(ns).values()) == pytest.approx([-5.0, -10.0])


# =========================================================================
# Views
# =========================================================================


class TestViews:
    """ArrayView and SeriesView access after run()."""

    def test_array_view(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [42.0]))
        sc.run()
        np.testing.assert_array_almost_equal(sc.array_view(h).value(), [42.0])

    def test_series_view(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1), ts(2)], [10.0, 20.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        sv = sc.series_view(hs)
        assert len(sv) == 2
        np.testing.assert_array_almost_equal(sv.values().flatten(), [10.0, 20.0])

    def test_array_view_repr_and_dtype(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [42.0]))
        sc.run()
        view = sc.array_view(h)
        assert view.dtype == np.dtype("float64")
        assert repr(view).startswith("ArrayView(")

    def test_series_view_repr(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1), ts(2)], [10.0, 20.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        view = sc.series_view(hs)
        assert repr(view).startswith("SeriesView(")

    def test_array_protocol(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0, 3.0]]))
        sc.run()
        arr = np.asarray(sc.array_view(h))
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_numpy_ufunc(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0, 3.0]]))
        sc.run()
        result = np.log(sc.array_view(h))
        np.testing.assert_array_almost_equal(result, np.log([1.0, 2.0, 3.0]))

    def test_array_getitem(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0, 3.0]]))
        sc.run()
        view = sc.array_view(h)
        assert view[0] == 1.0
        np.testing.assert_array_equal(view[1:3], [2.0, 3.0])

    def test_array_arithmetic(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0, 3.0]]))
        sc.run()
        view = sc.array_view(h)
        np.testing.assert_array_equal(view + 10, [11.0, 12.0, 13.0])
        np.testing.assert_array_equal(view * 2, [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(-view, [-1.0, -2.0, -3.0])

    def test_array_to_numpy(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(1)], [[1.0, 2.0, 3.0]]))
        sc.run()
        np.testing.assert_array_equal(sc.array_view(h).to_numpy(), [1.0, 2.0, 3.0])

    def test_series_timestamps_dtype(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        t = sc.series_view(hs).timestamps()
        assert t.dtype == np.dtype("datetime64[ns]")
        assert len(t) == 3

    def test_series_at(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        sv = sc.series_view(hs)
        np.testing.assert_array_almost_equal(sv.at(0), [10.0])
        np.testing.assert_array_almost_equal(sv.at(2), [30.0])
        np.testing.assert_array_almost_equal(sv.at(-1), [30.0])
        np.testing.assert_array_almost_equal(sv.at(-3), [10.0])

    def test_series_asof(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        sv = sc.series_view(hs)
        np.testing.assert_array_almost_equal(sv.asof(ts(200)), [20.0])  # type: ignore
        np.testing.assert_array_almost_equal(sv.asof(ts(250)), [20.0])  # type: ignore
        assert sv.asof(ts(50)) is None

    def test_series_to_numpy(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        t, vals = sc.series_view(hs).to_numpy()
        assert t.dtype == np.dtype("datetime64[ns]")
        assert len(t) == 3
        np.testing.assert_array_almost_equal(vals.flatten(), [10.0, 20.0, 30.0])

    def test_series_getitem(self) -> None:
        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        sv = sc.series_view(hs)
        np.testing.assert_array_almost_equal(sv[0], [10.0])
        np.testing.assert_array_almost_equal(sv[-1], [30.0])
        np.testing.assert_array_almost_equal(sv[1:3].flatten(), [20.0, 30.0])

    def test_series_to_series(self) -> None:
        import pandas as pd

        sc = Scenario()
        h = sc.add_source(ArraySource([ts(100), ts(200), ts(300)], [10.0, 20.0, 30.0]))
        hs = sc.add_operator(Record(h))
        sc.run()
        s = sc.series_view(hs).to_series()
        assert isinstance(s, pd.Series)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert len(s) == 3


# =========================================================================
# Scenario event loop invariants
# =========================================================================


class TestScenario:
    """Event ordering, coalescing, and event-flow invariants."""

    def test_same_timestamp_updates_are_coalesced(self) -> None:
        source_a = ArraySource([ts(1), ts(1)], [1.0, 2.0])
        source_b = ArraySource([ts(1)], [3.0])
        sc = Scenario()
        a = sc.add_source(source_a)
        b = sc.add_source(source_b)
        hc = sc.add_operator(Add(a, b))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).timestamps()) == [ts(1)]
        assert list(sc.series_view(hs).values()) == pytest.approx([5.0])

    def test_run_rejects_decreasing_source_timestamps(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            ArraySource([ts(2), ts(1)], [1.0, 2.0])

    def test_output_timestamps_strictly_increasing(self) -> None:
        rng = np.random.default_rng(42)
        sources = []
        for i in range(4):
            n = int(rng.integers(5, 15))
            raw_ts = np.sort(rng.choice(200, size=n, replace=False))
            vals = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
            sources.append(ArraySource(timestamps=raw_ts, values=vals, name=f"src_{i}"))
        sc = Scenario()
        obs = [sc.add_source(src) for src in sources]
        result_h = sc.add_operator(Add(obs[0], obs[1]))
        for o in obs[2:]:
            result_h = sc.add_operator(Add(result_h, o))
        hs = sc.add_operator(Record(result_h))
        sc.run()
        ts_arr = sc.series_view(hs).timestamps()
        assert len(ts_arr) > 0
        for i in range(1, len(ts_arr)):
            assert ts_arr[i] > ts_arr[i - 1]

    def test_two_random_sources_add_matches_simulation(self) -> None:
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
            if t in events_a:
                last_a = events_a[t]
            if t in events_b:
                last_b = events_b[t]
            exp_ts.append(np.datetime64(t, "ns"))
            exp_vals.append(last_a + last_b)
        sc = Scenario()
        a = sc.add_source(ArraySource(raw_ts_a, vals_a.astype(np.float64)))
        b = sc.add_source(ArraySource(raw_ts_b, vals_b.astype(np.float64)))
        hc = sc.add_operator(Add(a, b))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).timestamps()) == exp_ts
        np.testing.assert_array_almost_equal(list(sc.series_view(hs).values()), exp_vals)

    def test_interleaved_sources_trigger_at_every_timestamp(self) -> None:
        ts_a = np.array([1, 3, 5, 7, 9], dtype="datetime64[ns]")
        ts_b = np.array([2, 4, 6, 8, 10], dtype="datetime64[ns]")
        sc = Scenario()
        a = sc.add_source(ArraySource(ts_a, [1.0, 2.0, 3.0, 4.0, 5.0]))
        b = sc.add_source(ArraySource(ts_b, [10.0, 20.0, 30.0, 40.0, 50.0]))
        hc = sc.add_operator(Add(a, b))
        hs = sc.add_operator(Record(hc))
        sc.run()
        expected = [1.0, 11.0, 12.0, 22.0, 23.0, 33.0, 34.0, 44.0, 45.0, 55.0]
        assert list(sc.series_view(hs).values()) == pytest.approx(expected)

    def test_run_updates_only_affected_downstream(self) -> None:
        sc = Scenario()
        a = sc.add_source(ArraySource([ts(1), ts(2)], [1.0, 4.0]))
        b = sc.add_source(ArraySource([ts(1)], [2.0]))
        c = sc.add_source(ArraySource([ts(1), ts(3)], [10.0, 5.0]))
        sum_h = sc.add_operator(Add(a, b))
        scaled_h = sc.add_operator(Multiply(sum_h, c))
        neg_h = sc.add_operator(Negate(scaled_h))
        sum_s = sc.add_operator(Record(sum_h))
        scaled_s = sc.add_operator(Record(scaled_h))
        neg_s = sc.add_operator(Record(neg_h))
        sc.run()
        assert list(sc.series_view(sum_s).timestamps()) == [ts(1), ts(2)]
        assert list(sc.series_view(scaled_s).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(sum_s).values()) == pytest.approx([3.0, 6.0])
        assert list(sc.series_view(scaled_s).values()) == pytest.approx([30.0, 60.0, 30.0])
        assert list(sc.series_view(neg_s).values()) == pytest.approx([-30.0, -60.0, -30.0])

    def test_same_source_added_twice(self) -> None:
        source = ArraySource([ts(1), ts(2), ts(3)], [10.0, 20.0, 30.0])
        sc = Scenario()
        o1 = sc.add_source(source)
        o2 = sc.add_source(source)
        assert o1 is not o2
        hc = sc.add_operator(Add(o1, o2))
        hs = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hs).values()) == pytest.approx([20.0, 40.0, 60.0])

    def test_run_is_repeatable(self) -> None:
        rng = np.random.default_rng(7)
        srcs = []
        for i in range(3):
            n = int(rng.integers(8, 15))
            raw_ts = np.sort(rng.choice(100, size=n, replace=False))
            vals = rng.uniform(0.0, 10.0, size=n).astype(np.float64)
            srcs.append(ArraySource(timestamps=raw_ts, values=vals))

        def build_and_run():
            sc = Scenario()
            obs = [sc.add_source(src) for src in srcs]
            h = sc.add_operator(Add(obs[0], obs[1]))
            h = sc.add_operator(Add(h, obs[2]))
            hs = sc.add_operator(Record(h))
            sc.run()
            return list(sc.series_view(hs).timestamps()), list(sc.series_view(hs).values())

        ts1, v1 = build_and_run()
        ts2, v2 = build_and_run()
        assert ts1 == ts2
        np.testing.assert_array_equal(v1, v2)


# =========================================================================
# Mixed Rust + Python sources and operators
# =========================================================================


class TestMixedSourcesAndOperators:
    """Scenarios with mixed native and Python sources/operators."""

    def test_native_and_py_sources_add(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(2), ts(3)], [10.0, 20.0, 30.0]))
        hb = sc.add_source(
            IterSource(
                [(ts(1), 100.0), (ts(2), 200.0), (ts(3), 300.0)],
                shape=(),
                dtype=np.float64,
            )
        )
        hc = sc.add_operator(Add(ha, hb))
        hr = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hr).values()) == pytest.approx([110.0, 220.0, 330.0])

    def test_native_and_py_sources_interleaved(self) -> None:
        sc = Scenario()
        ha = sc.add_source(ArraySource([ts(1), ts(3)], [10.0, 30.0]))
        hb = sc.add_source(
            IterSource(
                [(ts(2), 20.0), (ts(3), 40.0)],
                shape=(),
                dtype=np.float64,
            )
        )
        hc = sc.add_operator(Add(ha, hb))
        hr = sc.add_operator(Record(hc))
        sc.run()
        # ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert list(sc.series_view(hr).timestamps()) == [ts(1), ts(2), ts(3)]
        assert list(sc.series_view(hr).values()) == pytest.approx([10.0, 30.0, 70.0])

    def test_py_source_native_op_py_op_chain(self) -> None:
        sc = Scenario()
        h = sc.add_source(
            IterSource(
                [(ts(i), float(i)) for i in range(1, 8)],
                shape=(),
                dtype=np.float64,
            )
        )
        c = sc.add_const(np.array(10.0, dtype=np.float64))
        hs = sc.add_operator(Multiply(h, c))
        hf = sc.add_operator(Filter(hs, lambda v: float(v.flat[0]) >= 50.0))
        hr = sc.add_operator(Record(hf))
        sc.run()
        assert list(sc.series_view(hr).values()) == pytest.approx([50.0, 60.0, 70.0])

    def test_multiple_py_sources(self) -> None:
        sc = Scenario()
        ha = sc.add_source(
            IterSource(
                [(ts(1), 1.0), (ts(2), 2.0)],
                shape=(),
                dtype=np.float64,
            )
        )
        hb = sc.add_source(
            IterSource(
                [(ts(1), 10.0), (ts(2), 20.0)],
                shape=(),
                dtype=np.float64,
            )
        )
        hc = sc.add_operator(Add(ha, hb))
        hr = sc.add_operator(Record(hc))
        sc.run()
        assert list(sc.series_view(hr).values()) == pytest.approx([11.0, 22.0])

    def test_py_source_vector_values(self) -> None:
        sc = Scenario()
        ha = sc.add_source(
            IterSource(
                [(ts(1), np.array([1.0, 2.0]))],
                shape=(2,),
                dtype=np.float64,
            )
        )
        hb = sc.add_source(
            IterSource(
                [(ts(1), np.array([10.0, 20.0]))],
                shape=(2,),
                dtype=np.float64,
            )
        )
        hc = sc.add_operator(Add(ha, hb))
        hr = sc.add_operator(Record(hc))
        sc.run()
        np.testing.assert_array_almost_equal(sc.series_view(hr).values().flatten(), [11.0, 22.0])

    def test_py_source_with_native_operator(self) -> None:
        sc = Scenario()
        h = sc.add_source(
            IterSource(
                [(ts(i), float(i)) for i in range(1, 11)],
                shape=(),
                dtype=np.float64,
            )
        )
        c = sc.add_const(np.array(2.0, dtype=np.float64))
        hs = sc.add_operator(Multiply(h, c))
        hr = sc.add_operator(Record(hs))
        sc.run()
        expected = [float(i) * 2.0 for i in range(1, 11)]
        assert list(sc.series_view(hr).values()) == pytest.approx(expected)

    def test_py_source_with_py_operator(self) -> None:
        """Python source + Python operator (Filter): the former deadlock case."""
        sc = Scenario()
        h = sc.add_source(
            IterSource(
                [(ts(i), float(i)) for i in range(1, 6)],
                shape=(),
                dtype=np.float64,
            )
        )
        hf = sc.add_operator(Filter(h, lambda v: float(v.flat[0]) > 3.0))
        hr = sc.add_operator(Record(hf))
        sc.run()
        assert list(sc.series_view(hr).values()) == pytest.approx([4.0, 5.0])
