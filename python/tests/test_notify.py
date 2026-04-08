"""Tests for the Notify mechanism — both native (ForwardAdjust) and Python operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tradingflow import Scenario, Operator, Notify
from tradingflow.sources import ArraySource, IterSource
from tradingflow.operators import Record
from tradingflow.operators.stocks import ForwardAdjust
from tradingflow.types import Array, Series, Handle, NodeKind


def ts(i: int) -> np.datetime64:
    return np.datetime64(i, "ns")


# =========================================================================
# ForwardAdjust (native operator using Notify)
# =========================================================================


class TestForwardAdjust:
    """Native ForwardAdjust operator, which uses Notify.input_produced()."""

    def test_no_dividends(self) -> None:
        """Adjusted close equals raw close when there are no dividends."""
        sc = Scenario()
        prices = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3)],
            values=np.array([10.0, 11.0, 12.0]),
        ))
        divs = sc.add_source(IterSource(
            iterable=[],
            shape=(2,),
            dtype=np.float64,
            initial=np.array([0.0, 0.0]),
        ))
        adj = sc.add_operator(ForwardAdjust(prices, divs))
        adj_s = sc.add_operator(Record(adj))
        sc.run()
        vals = list(sc.series_view(adj_s).values())
        assert vals == pytest.approx([10.0, 11.0, 12.0])

    def test_cash_dividend(self) -> None:
        """Cash dividend adjusts prices forward."""
        sc = Scenario()
        prices = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2)],
            values=np.array([10.0, 9.5]),
        ))
        divs = sc.add_source(IterSource(
            iterable=[(ts(2), np.array([0.0, 0.5]))],
            shape=(2,),
            dtype=np.float64,
            initial=np.array([0.0, 0.0]),
        ))
        adj = sc.add_operator(ForwardAdjust(prices, divs))
        adj_s = sc.add_operator(Record(adj))
        sc.run()
        vals = list(sc.series_view(adj_s).values())
        assert vals[0] == pytest.approx(10.0)
        factor = 1.0 + 0.5 / 9.5
        assert vals[1] == pytest.approx(9.5 * factor)

    def test_share_dividend(self) -> None:
        """Share dividend (bonus shares) adjusts prices forward."""
        sc = Scenario()
        prices = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2)],
            values=np.array([20.0, 18.0]),
        ))
        divs = sc.add_source(IterSource(
            iterable=[(ts(2), np.array([0.1, 0.0]))],
            shape=(2,),
            dtype=np.float64,
            initial=np.array([0.0, 0.0]),
        ))
        adj = sc.add_operator(ForwardAdjust(prices, divs))
        adj_s = sc.add_operator(Record(adj))
        sc.run()
        vals = list(sc.series_view(adj_s).values())
        assert vals[0] == pytest.approx(20.0)
        assert vals[1] == pytest.approx(18.0 * 1.1)

    def test_cumulative_dividends(self) -> None:
        """Two successive dividends compound."""
        sc = Scenario()
        prices = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3), ts(4)],
            values=np.array([100.0, 98.0, 99.0, 98.0]),
        ))
        divs = sc.add_source(IterSource(
            iterable=[
                (ts(2), np.array([0.0, 2.0])),
                (ts(4), np.array([0.0, 1.0])),
            ],
            shape=(2,),
            dtype=np.float64,
            initial=np.array([0.0, 0.0]),
        ))
        adj = sc.add_operator(ForwardAdjust(prices, divs))
        adj_s = sc.add_operator(Record(adj))
        sc.run()
        vals = list(sc.series_view(adj_s).values())
        f1 = 1.0 + 2.0 / (100.0 - 2.0)
        assert vals[2] == pytest.approx(99.0 * f1)
        f2 = f1 * (1.0 + 1.0 / (99.0 - 1.0))
        assert vals[3] == pytest.approx(98.0 * f2)


# =========================================================================
# Python operator using Notify — message-queue semantics
# =========================================================================


class SelectiveRecorder(
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]]],
        Handle[Series[np.float64]],
        None,
    ]
):
    """Records input 1 into the output Series only when input 1 produces.

    Input 0 is a "background" source that fires frequently but whose
    values are ignored.  Input 1 is a sparse "message" source.  The
    output Series should contain exactly the same elements as
    ``Record(input_1)`` — regardless of how often input 0 fires.
    """

    def __init__(self, background: Handle, messages: Handle) -> None:
        super().__init__(
            inputs=(background, messages),
            kind=NodeKind.SERIES,
            dtype=messages.dtype,
            shape=messages.shape,
        )

    def init(self, inputs: tuple, timestamp: int) -> None:
        return None

    @staticmethod
    def compute(
        state: None,
        inputs: tuple,
        output: Any,
        timestamp: int,
        notify: Any,
    ) -> bool:
        if notify.input_produced()[1]:
            output.push(timestamp, inputs[1].value())
            return True
        return False


class TestPythonNotify:
    """Python-implemented operator that uses Notify for selective recording."""

    def test_selective_recorder_matches_record(self) -> None:
        """SelectiveRecorder output equals Record applied to the message source."""
        sc = Scenario()

        # Background fires every tick.
        background = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3), ts(4), ts(5)],
            values=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        ))

        # Messages fire only at ts=2 and ts=4.
        messages = sc.add_source(IterSource(
            iterable=[(ts(2), np.array(10.0)), (ts(4), np.array(20.0))],
            shape=(),
            dtype=np.float64,
            initial=np.array(0.0),
        ))

        # SelectiveRecorder: should only record when messages fires.
        sel = sc.add_operator(SelectiveRecorder(background, messages))

        # Reference: plain Record on messages.
        msg_series = sc.add_operator(Record(messages))

        sc.run()

        sel_view = sc.series_view(sel)
        ref_view = sc.series_view(msg_series)

        # Same number of elements.
        assert len(sel_view) == len(ref_view)

        # Same timestamps.
        np.testing.assert_array_equal(
            sel_view.timestamps(),
            ref_view.timestamps(),
        )

        # Same values.
        np.testing.assert_array_almost_equal(
            sel_view.values(),
            ref_view.values(),
        )

    def test_selective_recorder_no_messages(self) -> None:
        """When the message source never fires, the output Series is empty."""
        sc = Scenario()

        background = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3)],
            values=np.array([1.0, 2.0, 3.0]),
        ))
        messages = sc.add_source(IterSource(
            iterable=[],
            shape=(),
            dtype=np.float64,
            initial=np.array(0.0),
        ))

        sel = sc.add_operator(SelectiveRecorder(background, messages))
        sc.run()

        assert len(sc.series_view(sel)) == 0

    def test_selective_recorder_coalesced_timestamps(self) -> None:
        """When both sources fire at the same timestamp, the message is recorded."""
        sc = Scenario()

        # Both fire at every timestamp.
        background = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3)],
            values=np.array([100.0, 200.0, 300.0]),
        ))
        messages = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3)],
            values=np.array([10.0, 20.0, 30.0]),
        ))

        sel = sc.add_operator(SelectiveRecorder(background, messages))
        sc.run()

        sel_view = sc.series_view(sel)
        assert len(sel_view) == 3
        np.testing.assert_array_almost_equal(
            sel_view.values().flatten(),
            [10.0, 20.0, 30.0],
        )
