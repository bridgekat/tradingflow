"""Tests for the Rust bridge (NativeScenario) interop.

These tests exercise the Rust runtime directly via the ``_native`` C extension,
without the Python-level Scenario wrapper, to validate the bridge layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow._native import NativeScenario


class TestBridgeNativeOperator:
    """Native operator registration via add_native_operator."""

    def _make_source(self, sc: NativeScenario, timestamps: list[int], values: list[float]) -> int:
        """Helper: register a channel source, push historical events, close."""
        idx, hist, live = sc.add_source_raw([], "float64")
        for ts, val in zip(timestamps, values):
            hist.send(ts, np.array([val], dtype=np.float64))
        hist.close()
        live.close()
        return idx

    def test_add_and_record(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1, 2], [10.0, 20.0])
        hb = self._make_source(sc, [1, 2], [1.0, 2.0])
        hc = sc.add_native_operator("add", "float64", [ha, hb], [], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        assert sc.series_len(hs) == 2
        np.testing.assert_array_almost_equal(
            np.asarray(sc.series_values(hs)),
            [11.0, 22.0],
        )

    def test_subtract(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1], [10.0])
        hb = self._make_source(sc, [1], [3.0])
        hc = sc.add_native_operator("subtract", "float64", [ha, hb], [], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [7.0])

    def test_multiply(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1], [4.0])
        hb = self._make_source(sc, [1], [5.0])
        hc = sc.add_native_operator("multiply", "float64", [ha, hb], [], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [20.0])

    def test_negate(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1, 2], [10.0, -5.0])
        hc = sc.add_native_operator("negate", "float64", [ha], [], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [-10.0, 5.0])

    def test_chained_operators(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1, 2], [2.0, 5.0])
        hb = self._make_source(sc, [1, 2], [3.0, 10.0])
        hab = sc.add_native_operator("add", "float64", [ha, hb], [], {})
        hout = sc.add_native_operator("multiply", "float64", [hab, ha], [], {})
        hs = sc.add_native_operator("record", "float64", [hout], [], {})
        sc.run()
        # (2+3)*2=10, (5+10)*5=75
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [10.0, 75.0])

    def test_interleaved_sources(self) -> None:
        sc = NativeScenario()
        ha = self._make_source(sc, [1, 3], [10.0, 30.0])
        hb = self._make_source(sc, [2, 3], [20.0, 40.0])
        hc = sc.add_native_operator("add", "float64", [ha, hb], [], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        # ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [10.0, 30.0, 70.0])


class TestBridgeStrided:
    """Tests with vector-valued (strided) arrays."""

    def _make_strided_source(self, sc: NativeScenario, timestamps: list[int], values: list[list[float]]) -> int:
        idx, hist, live = sc.add_source_raw([len(values[0])], "float64")
        for ts, val in zip(timestamps, values):
            hist.send(ts, np.array(val, dtype=np.float64))
        hist.close()
        live.close()
        return idx

    def test_strided_add(self) -> None:
        sc = NativeScenario()
        ha = self._make_strided_source(sc, [1], [[1.0, 2.0]])
        hb = self._make_strided_source(sc, [1], [[10.0, 20.0]])
        hc = sc.add_native_operator("add", "float64", [ha, hb], [2], {})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [11.0, 22.0])

    def test_select(self) -> None:
        sc = NativeScenario()
        ha = self._make_strided_source(sc, [1], [[10.0, 20.0, 30.0]])
        hc = sc.add_native_operator("select", "float64", [ha], [2], {"indices": [0, 2]})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [10.0, 30.0])

    def test_concat(self) -> None:
        sc = NativeScenario()
        ha = self._make_strided_source(sc, [1], [[1.0, 2.0]])
        hb = self._make_strided_source(sc, [1], [[3.0, 4.0]])
        hc = sc.add_native_operator("concat", "float64", [ha, hb], [4], {"axis": 0})
        hs = sc.add_native_operator("record", "float64", [hc], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [1.0, 2.0, 3.0, 4.0])


class TestBridgePyOperator:
    """Python operator registration via add_py_operator."""

    def test_py_operator_add_const(self) -> None:
        """Stateless Python operator that adds a constant."""
        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([10.0], dtype=np.float64))
        hist.send(2, np.array([20.0], dtype=np.float64))
        hist.close()
        live.close()

        class AddConst:
            def compute(self, timestamp, inputs, output, state):
                val = inputs[0].value()
                output.write(val + 5.0)
                return True, state

        op_idx = sc.add_py_operator(
            [idx],
            [("array", "float64")],
            ("array", "float64"),
            [],
            AddConst(),
            None,
        )
        hs = sc.add_native_operator("record", "float64", [op_idx], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [15.0, 25.0])

    def test_py_operator_stateful(self) -> None:
        """Stateful Python operator that computes running sum."""
        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([10.0], dtype=np.float64))
        hist.send(2, np.array([20.0], dtype=np.float64))
        hist.send(3, np.array([30.0], dtype=np.float64))
        hist.close()
        live.close()

        class RunningSum:
            def compute(self, timestamp, inputs, output, state):
                val = float(inputs[0].value().flat[0])
                new_state = (state or 0.0) + val
                output.write(np.array([new_state], dtype=np.float64))
                return True, new_state

        op_idx = sc.add_py_operator(
            [idx],
            [("array", "float64")],
            ("array", "float64"),
            [],
            RunningSum(),
            None,
        )
        hs = sc.add_native_operator("record", "float64", [op_idx], [], {})
        sc.run()
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [10.0, 30.0, 60.0])

    def test_py_operator_filter(self) -> None:
        """Python operator that skips propagation by returning False."""
        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([1.0], dtype=np.float64))
        hist.send(2, np.array([5.0], dtype=np.float64))
        hist.send(3, np.array([2.0], dtype=np.float64))
        hist.send(4, np.array([10.0], dtype=np.float64))
        hist.close()
        live.close()

        class FilterAbove3:
            def compute(self, timestamp, inputs, output, state):
                val = float(inputs[0].value().flat[0])
                if val > 3.0:
                    output.write(np.array([val], dtype=np.float64))
                    return True, state
                return False, state

        op_idx = sc.add_py_operator(
            [idx],
            [("array", "float64")],
            ("array", "float64"),
            [],
            FilterAbove3(),
            None,
        )
        hs = sc.add_native_operator("record", "float64", [op_idx], [], {})
        sc.run()
        assert sc.series_len(hs) == 2
        np.testing.assert_array_almost_equal(np.asarray(sc.series_values(hs)), [5.0, 10.0])


class TestBridgeGetView:
    """Test get_view for raw native views."""

    def test_array_view(self) -> None:
        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([42.0], dtype=np.float64))
        hist.close()
        live.close()
        sc.run()
        view = sc.get_view(idx)
        np.testing.assert_array_almost_equal(view.value(), [42.0])

    def test_series_view(self) -> None:
        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([10.0], dtype=np.float64))
        hist.send(2, np.array([20.0], dtype=np.float64))
        hist.close()
        live.close()
        hs = sc.add_native_operator("record", "float64", [idx], [], {})
        sc.run()
        view = sc.get_view(hs)
        assert len(view) == 2
        np.testing.assert_array_almost_equal(np.asarray(view.values()).flatten(), [10.0, 20.0])


class TestPythonViewWrappers:
    """Test the Python ArrayView/SeriesView wrappers over native views."""

    def test_array_view_wrapper(self) -> None:
        from tradingflow.views import ArrayView

        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([42.0], dtype=np.float64))
        hist.close()
        live.close()
        sc.run()
        native_view = sc.get_view(idx)
        wrapper = ArrayView(native_view)
        np.testing.assert_array_almost_equal(wrapper.value(), [42.0])
        assert wrapper.dtype == np.dtype("float64")
        assert repr(wrapper).startswith("ArrayView(")

    def test_series_view_wrapper(self) -> None:
        from tradingflow.views import SeriesView

        sc = NativeScenario()
        idx, hist, live = sc.add_source_raw([], "float64")
        hist.send(1, np.array([10.0], dtype=np.float64))
        hist.send(2, np.array([20.0], dtype=np.float64))
        hist.close()
        live.close()
        hs = sc.add_native_operator("record", "float64", [idx], [], {})
        sc.run()
        native_view = sc.get_view(hs)
        wrapper = SeriesView(native_view)
        assert len(wrapper) == 2
        np.testing.assert_array_almost_equal(np.asarray(wrapper.values()).flatten(), [10.0, 20.0])
        assert repr(wrapper).startswith("SeriesView(")
