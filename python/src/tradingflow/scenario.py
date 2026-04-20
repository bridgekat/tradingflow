"""Scenario runtime -- thin Python wrapper around the Rust native backend."""

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

    Sources and operators are registered via `add_source` and
    `add_operator`, each returning a `Handle`. To record history,
    use the `record` operator explicitly. `run` executes the POCQ
    event loop.
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
        # Unit nodes carry no value; report `void` dtype on the handle.
        dtype = np.dtype("void") if source.kind == NodeKind.UNIT else np.dtype(source.dtype)
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
        return Handle(idx, operator.kind, operator.dtype, operator.shape)

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
        """Execute the POCQ event loop.

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
