"""Tests for the Notify mechanism — both native (ForwardAdjust) and Python operators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tradingflow import Scenario, Operator, Notify
from tradingflow.sources import ArraySource, IterSource
from tradingflow.operators import Record
from tradingflow.operators.stocks import ForwardAdjust
from tradingflow.types import Array, Handle


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
# Python operator using Notify
# =========================================================================


class ConditionalAccumulator(
    Operator[
        tuple[Handle[Array[np.float64]], Handle[Array[np.float64]]],
        Handle[Array[np.float64]],
        float,
    ]
):
    """Accumulates input 0 only when input 1 updates (uses Notify)."""

    def __init__(self, data: Handle, trigger: Handle) -> None:
        super().__init__(inputs=(data, trigger), shape=(), dtype=np.float64)

    def init_state(self) -> float:
        return 0.0

    def compute(
        self,
        timestamp: int,
        inputs: tuple,
        output: Any,
        state: float,
        notify: Any,
    ) -> tuple[bool, float]:
        if notify.input_produced(1):
            state += float(inputs[0].value().flat[0])
        output.write(np.array(state))
        return True, state


class TestPythonNotify:
    """Python-implemented operator that reads Notify."""

    def test_accumulates_only_on_trigger(self) -> None:
        """ConditionalAccumulator only adds input 0 when input 1 fires."""
        sc = Scenario()
        data = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2), ts(3), ts(4)],
            values=np.array([10.0, 20.0, 30.0, 40.0]),
        ))
        trigger = sc.add_source(IterSource(
            iterable=[(ts(2), np.array(1.0)), (ts(4), np.array(1.0))],
            shape=(),
            dtype=np.float64,
            initial=np.array(0.0),
        ))
        acc = sc.add_operator(ConditionalAccumulator(data, trigger))
        acc_s = sc.add_operator(Record(acc))
        sc.run()

        vals = list(sc.series_view(acc_s).values())
        # ts=1: only data fires → state stays 0
        assert vals[0] == pytest.approx(0.0)
        # ts=2: both fire → state = 0 + 20 = 20
        assert vals[1] == pytest.approx(20.0)
        # ts=3: only data fires → state stays 20
        assert vals[2] == pytest.approx(20.0)
        # ts=4: both fire → state = 20 + 40 = 60
        assert vals[3] == pytest.approx(60.0)

    def test_input_produced_false_when_not_updated(self) -> None:
        """input_produced returns False for inputs that didn't fire."""
        sc = Scenario()
        data = sc.add_source(ArraySource(
            timestamps=[ts(1), ts(2)],
            values=np.array([1.0, 2.0]),
        ))
        trigger = sc.add_source(IterSource(
            iterable=[(ts(2), np.array(1.0))],
            shape=(),
            dtype=np.float64,
            initial=np.array(0.0),
        ))
        acc = sc.add_operator(ConditionalAccumulator(data, trigger))
        acc_s = sc.add_operator(Record(acc))
        sc.run()

        vals = list(sc.series_view(acc_s).values())
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(2.0)
