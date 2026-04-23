"""Scenario — the Python entry point to the Rust computation graph.

A [`Scenario`][tradingflow.scenario.Scenario] is the single object you interact
with to build and run a strategy.  It owns the directed acyclic
computation graph plus the event loop that drives it.

Typical usage has three phases:

1. **Construct** — `sc = Scenario()`.
2. **Populate** — call `sc.add_source(...)` and `sc.add_operator(...)`,
   passing the handles returned by earlier calls as inputs to later
   operators.  Every call returns a typed
   [`Handle`][tradingflow.data.types.Handle] that encodes the new node's value
   kind (array vs. series), shape, and dtype.
3. **Run** — `sc.run()` drains every registered source in timestamp
   order, propagates each flush batch through the graph, and returns
   when every source is exhausted.  After it returns, use
   `sc.array_view(handle)` or `sc.series_view(handle)` to inspect the
   final state of any node.

This module is a thin Python wrapper over the Rust native backend;
the bulk of the work happens inside the Rust core.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tradingflow._native import NativeArrayView, NativeSeriesView, NativeScenario
from . import Array, Series, operators
from .operator import Operator, NativeOperator
from .source import Source, NativeSource
from .data.types import Handle, NodeKind
from .data.views import ArrayView, SeriesView


class Scenario:
    """A directed acyclic graph of sources and operators.

    Sources and operators are registered via
    [`add_source`][tradingflow.scenario.Scenario.add_source] and
    [`add_operator`][tradingflow.scenario.Scenario.add_operator], each returning
    a [`Handle`][tradingflow.data.types.Handle].  Node output values are not
    historised automatically — attach a
    [`Record`][tradingflow.operators.record.Record] operator where a time
    series is required.

    [`run`][tradingflow.scenario.Scenario.run] drives the async event loop: it
    drains every source's historical and live channels in timestamp
    order, coalesces events that share the same timestamp into a single
    flush batch, and propagates the batch through the graph before
    advancing to the next timestamp.  Within a batch, each operator's
    `produced` mask reports which of its inputs actually produced this
    cycle (see the "Notification semantics" section in
    [`tradingflow`][tradingflow]).
    """

    __slots__ = ("_native",)

    def __init__(self) -> None:
        self._native = NativeScenario()

    def array_view(self, handle: Handle[Array[Any]]) -> ArrayView:
        """Get an ArrayView for an Array node."""
        inner = self._native.view(handle.index)
        assert isinstance(inner, NativeArrayView)
        return ArrayView(inner)

    def series_view(self, handle: Handle[Series[Any]]) -> SeriesView:
        """Get a SeriesView for a Series node."""
        inner = self._native.view(handle.index)
        assert isinstance(inner, NativeSeriesView)
        return SeriesView(inner)

    def add_const(self, value: np.ndarray) -> Handle:
        """Register a constant node with an initial value.

        Shorthand for `add_operator(Const(value))`.

        Parameters
        ----------
        value
            Initial value.
        """
        return self.add_operator(operators.Const(value))

    def add_source(self, source: Source | NativeSource) -> Handle:
        """Register a source and return a handle to its output node.

        Polymorphic dispatch via `source._register`: native and Python
        sources share a single code path.
        """
        idx = source._register(self._native)
        # Unit nodes carry no value; report `None` dtype on the handle.
        dtype = None if source.kind == NodeKind.UNIT else np.dtype(source.dtype)
        return Handle(idx, source.kind, dtype, source.shape)

    def add_operator(
        self,
        operator: NativeOperator | Operator,
    ) -> Handle:
        """Register an operator and return a handle to its output node.

        Polymorphic dispatch via `operator._register`: native and Python
        operators share a single code path.

        Parameters
        ----------
        operator
            The operator to register (native or Python).
        """
        input_indices = [inp.index for inp in operator.inputs]
        idx = operator._register(self._native, input_indices)
        # Unit outputs carry no value; report `None` dtype on the handle.
        dtype = None if operator.kind == NodeKind.UNIT else np.dtype(operator.dtype)
        return Handle(idx, operator.kind, dtype, operator.shape)

    def estimated_event_count(self) -> int | None:
        """Sum of estimated event counts across all sources.

        Returns `None` if any registered source is unable to produce an
        estimate (e.g. live / unbounded sources).  CSV-backed sources
        estimate rows by sampling the first kilobytes of the file, so the
        result is approximate.
        """
        return self._native.estimated_event_count()

    def run(
        self,
        on_flush: Callable[[int, int, int | None], Any] | None = None,
    ) -> None:
        """Execute the event loop.

        Python sources are driven by Rust-side async tasks that iterate
        the source's async iterators via a background asyncio event loop.
        The GIL is acquired briefly per event, preventing deadlocks with
        Python operators.

        Parameters
        ----------
        on_flush
            Optional callback invoked after each timestamp batch is
            flushed.  Receives `(timestamp_ns, events_so_far,
            total_estimate)` where `timestamp_ns` is the batch
            timestamp in TAI ns since the PTP epoch (re-view as
            `datetime64[ns]` to display), `events_so_far` is the
            running count of consumed source events, and
            `total_estimate` is the aggregate estimate (re-read per
            flush) or `None` when any source cannot estimate.
        """
        self._native.run(on_flush)
