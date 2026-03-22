"""Tests for the Rust bridge (NativeScenario) interop.

These tests exercise the Rust runtime directly via the `_native` C extension,
without the Python-level Scenario wrapper, to validate the bridge layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tradingflow._native import (
    NativeScenario,
    add,
    subtract,
    multiply,
    negate,
    select,
    concat,
)


class TestBridgeBasic:
    """Basic source + operator tests via the Rust bridge."""

    def test_add_source_and_record(self) -> None:
        """Register an array source, record it, and read back values."""
        sc = NativeScenario()
        timestamps = np.array([1, 2, 3], dtype=np.int64)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        node = sc.add_source([], "float64", timestamps, values.tobytes(), 1)
        series_node = sc.record(node)
        sc.run()

        sv = sc.store_view(series_node)
        assert len(sv) == 3
        np.testing.assert_array_equal(sv.index, [1, 2, 3])
        np.testing.assert_array_almost_equal(sv.values.flatten(), [10.0, 20.0, 30.0])

    def test_add_operator(self) -> None:
        """Register two sources, add them, and check the result."""
        sc = NativeScenario()
        ts = np.array([1, 2], dtype=np.int64)
        a_vals = np.array([10.0, 20.0], dtype=np.float64)
        b_vals = np.array([1.0, 2.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, a_vals.tobytes(), 1)
        hb = sc.add_source([], "float64", ts, b_vals.tobytes(), 1)

        add_handle = add("float64")
        ho = sc.register_handle_operator(add_handle, [ha, hb])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        assert len(sv) == 2
        np.testing.assert_array_almost_equal(sv.values.flatten(), [11.0, 22.0])

    def test_subtract_operator(self) -> None:
        """Subtraction operator works correctly."""
        sc = NativeScenario()
        ts = np.array([1], dtype=np.int64)
        a = np.array([10.0], dtype=np.float64)
        b = np.array([3.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, a.tobytes(), 1)
        hb = sc.add_source([], "float64", ts, b.tobytes(), 1)

        sub_handle = subtract("float64")
        ho = sc.register_handle_operator(sub_handle, [ha, hb])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        np.testing.assert_array_almost_equal(sv.values.flatten(), [7.0])

    def test_negate_operator(self) -> None:
        """Unary negate operator."""
        sc = NativeScenario()
        ts = np.array([1, 2], dtype=np.int64)
        vals = np.array([5.0, -3.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)

        neg_handle = negate("float64")
        ho = sc.register_handle_operator(neg_handle, [ha])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        np.testing.assert_array_almost_equal(sv.values.flatten(), [-5.0, 3.0])


class TestBridgeChaining:
    """Operator chaining tests."""

    def test_chained_operators(self) -> None:
        """Chain: add(a, b) -> multiply(sum, a)."""
        sc = NativeScenario()
        ts = np.array([1, 2], dtype=np.int64)
        a = np.array([2.0, 5.0], dtype=np.float64)
        b = np.array([3.0, 10.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, a.tobytes(), 1)
        hb = sc.add_source([], "float64", ts, b.tobytes(), 1)

        add_h = add("float64")
        h_sum = sc.register_handle_operator(add_h, [ha, hb])

        mul_h = multiply("float64")
        h_out = sc.register_handle_operator(mul_h, [h_sum, ha])
        hs = sc.record(h_out)

        sc.run()

        sv = sc.store_view(hs)
        # ts=1: (2+3)*2=10, ts=2: (5+10)*5=75
        np.testing.assert_array_almost_equal(sv.values.flatten(), [10.0, 75.0])

    def test_interleaved_sources(self) -> None:
        """Two sources with different timestamps, coalesced at shared timestamps."""
        sc = NativeScenario()
        ts_a = np.array([1, 3], dtype=np.int64)
        ts_b = np.array([2, 3], dtype=np.int64)
        a = np.array([10.0, 30.0], dtype=np.float64)
        b = np.array([20.0, 40.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts_a, a.tobytes(), 1)
        hb = sc.add_source([], "float64", ts_b, b.tobytes(), 1)

        add_h = add("float64")
        ho = sc.register_handle_operator(add_h, [ha, hb])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        # ts=1: 10+0=10, ts=2: 10+20=30, ts=3: 30+40=70
        assert len(sv) == 3
        np.testing.assert_array_equal(sv.index, [1, 2, 3])
        np.testing.assert_array_almost_equal(sv.values.flatten(), [10.0, 30.0, 70.0])


class TestBridgeStrided:
    """Tests with multi-element (strided) arrays."""

    def test_strided_source(self) -> None:
        """Source with stride > 1 produces vector-valued elements."""
        sc = NativeScenario()
        ts = np.array([1, 2], dtype=np.int64)
        vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        ha = sc.add_source([2], "float64", ts, vals.tobytes(), 2)
        hs = sc.record(ha)

        sc.run()

        sv = sc.store_view(hs)
        assert len(sv) == 2
        assert sv.shape == (2,)
        np.testing.assert_array_almost_equal(sv.values, [[1.0, 2.0], [3.0, 4.0]])

    def test_strided_add(self) -> None:
        """Element-wise add on vector-valued sources."""
        sc = NativeScenario()
        ts = np.array([1], dtype=np.int64)
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        ha = sc.add_source([3], "float64", ts, a.tobytes(), 3)
        hb = sc.add_source([3], "float64", ts, b.tobytes(), 3)

        add_h = add("float64")
        ho = sc.register_handle_operator(add_h, [ha, hb])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        np.testing.assert_array_almost_equal(sv.values.flatten(), [11.0, 22.0, 33.0])

    def test_select_operator(self) -> None:
        """Select operator picks specific indices from a vector."""
        sc = NativeScenario()
        ts = np.array([1], dtype=np.int64)
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

        ha = sc.add_source([5], "float64", ts, vals.tobytes(), 5)

        sel_h = select("float64", [1, 3])
        ho = sc.register_handle_operator(sel_h, [ha])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        np.testing.assert_array_almost_equal(sv.values.flatten(), [20.0, 40.0])

    def test_concat_operator(self) -> None:
        """Concat operator joins multiple sources along axis 0."""
        sc = NativeScenario()
        ts = np.array([1], dtype=np.int64)
        a = np.array([1.0, 2.0], dtype=np.float64)
        b = np.array([3.0, 4.0], dtype=np.float64)

        ha = sc.add_source([2], "float64", ts, a.tobytes(), 2)
        hb = sc.add_source([2], "float64", ts, b.tobytes(), 2)

        cat_h = concat("float64", [2], 0)
        ho = sc.register_handle_operator(cat_h, [ha, hb])
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        np.testing.assert_array_almost_equal(sv.values.flatten(), [1.0, 2.0, 3.0, 4.0])


class TestBridgePyOperator:
    """Tests for Python operator interop.

    RESTRICTIONS on Python operators registered via ``add_py_operator``:

    1. The ``compute(inputs, state)`` callback MUST NOT call back into
       the ``NativeScenario`` instance (e.g. ``sc.store_view()``,
       ``sc.run()``, ``sc.record()``).  The scenario is mutably
       borrowed during ``run()``; re-entering it causes a panic
       (``RuntimeError: Already mutably borrowed``).

    2. Input data should be communicated via the ``py_inputs`` / ``py_state``
       objects passed at registration time, NOT by capturing the scenario.

    3. The callback runs under the GIL during Rust's synchronous ``flush()``.
       Long-running Python code in ``compute`` blocks the entire DAG.
    """

    def test_py_operator_add_const(self) -> None:
        """A Python operator that adds 100 to each input element.

        The operator reads the current input value via a pre-built StoreView
        captured in py_state BEFORE run() (when the scenario is not borrowed).
        """
        sc = NativeScenario()
        ts = np.array([1, 2], dtype=np.int64)
        vals = np.array([5.0, 10.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)

        # Capture a StoreView of the input BEFORE run().
        # This is safe because StoreView holds a raw pointer to the node
        # value, which remains valid throughout the scenario's lifetime.
        input_view = sc.store_view(ha)

        class AddConstOp:
            """Adds 100 to the input value."""

            def compute(self, inputs, state):
                # Read current input from the pre-captured StoreView.
                val = float(inputs.last.flatten()[0])
                result = np.array([val + 100.0], dtype=np.float64)
                return (result, state)

        default = np.array([0.0], dtype=np.float64)

        ho = sc.add_py_operator(
            [ha],  # input indices
            [],  # shape (scalar)
            "float64",  # dtype
            default.tobytes(),  # default value
            AddConstOp(),  # Python operator
            input_view,  # py_inputs: pre-captured view (NOT the scenario)
            None,  # py_state: unused
        )
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        assert len(sv) == 2
        np.testing.assert_array_almost_equal(sv.values.flatten(), [105.0, 110.0])

    def test_py_operator_stateful(self) -> None:
        """A Python operator that accumulates a running sum."""
        sc = NativeScenario()
        ts = np.array([1, 2, 3], dtype=np.int64)
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)
        input_view = sc.store_view(ha)

        class RunningSumOp:
            def compute(self, inputs, state):
                val = float(inputs.last.flatten()[0])
                total = (state or 0.0) + val
                return (np.array([total], dtype=np.float64), total)

        default = np.array([0.0], dtype=np.float64)

        ho = sc.add_py_operator(
            [ha],
            [],
            "float64",
            default.tobytes(),
            RunningSumOp(),
            input_view,
            None,
        )
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        assert len(sv) == 3
        # Running sum: 1, 1+2=3, 1+2+3=6
        np.testing.assert_array_almost_equal(sv.values.flatten(), [1.0, 3.0, 6.0])

    def test_py_operator_filter(self) -> None:
        """A Python operator that returns None to skip output."""
        sc = NativeScenario()
        ts = np.array([1, 2, 3, 4], dtype=np.int64)
        vals = np.array([1.0, 5.0, 2.0, 10.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)
        input_view = sc.store_view(ha)

        class FilterOp:
            def compute(self, inputs, state):
                val = float(inputs.last.flatten()[0])
                if val > 3.0:
                    return (np.array([val], dtype=np.float64), state)
                else:
                    return (None, state)

        default = np.array([0.0], dtype=np.float64)

        ho = sc.add_py_operator(
            [ha],
            [],
            "float64",
            default.tobytes(),
            FilterOp(),
            input_view,
            None,
        )
        hs = sc.record(ho)

        sc.run()

        sv = sc.store_view(hs)
        # Only values > 3.0 pass: 5.0 (ts=2), 10.0 (ts=4)
        assert len(sv) == 2
        np.testing.assert_array_almost_equal(sv.values.flatten(), [5.0, 10.0])


class TestBridgeStoreView:
    """Tests for StoreView accessors."""

    def test_array_node_view(self) -> None:
        """StoreView on an Array node returns current value."""
        sc = NativeScenario()
        ts = np.array([1], dtype=np.int64)
        vals = np.array([42.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)
        sc.run()

        sv = sc.store_view(ha)
        assert sv.last.flatten()[0] == pytest.approx(42.0)
        assert sv.dtype == np.dtype("float64")

    def test_series_node_view(self) -> None:
        """StoreView on a Series node returns full history."""
        sc = NativeScenario()
        ts = np.array([10, 20, 30], dtype=np.int64)
        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        ha = sc.add_source([], "float64", ts, vals.tobytes(), 1)
        hs = sc.record(ha)
        sc.run()

        sv = sc.store_view(hs)
        assert len(sv) == 3
        np.testing.assert_array_equal(sv.index, [10, 20, 30])
        np.testing.assert_array_almost_equal(sv.values.flatten(), [1.0, 2.0, 3.0])
        assert sv.last.flatten()[0] == pytest.approx(3.0)
        assert sv.shape == ()
