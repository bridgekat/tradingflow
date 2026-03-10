"""Scenario runtime for source-driven series/operator execution.

This module defines :class:`Scenario`, a specification of sources and
operators that form a directed acyclic graph of time series, and
:class:`_ScenarioState`, the per-run mutable state created fresh for
every :meth:`Scenario.run` invocation.

Point-of-coherency queue (POCQ)
-------------------------------
Each source exposes a ``(historical, live)`` iterator pair via
:meth:`~src.source.Source.subscribe`.  The runtime converts all incoming items
into ``(timestamp, series, value)`` events and accumulates them in the POCQ.

* **Historical constraint** – before advancing the POCQ, every active
  historical iterator must have a pending event ready.  This prevents an
  in-flight historical iterator from later producing a timestamp smaller than
  one already committed.
* **Live iterators** are exempt from this constraint: their timestamps are
  wall-clock instants and are therefore always >= any historical timestamp
  and any existing live timestamp.
* The POCQ accumulates events sharing the same timestamp.  When an event with
  a strictly larger timestamp arrives, all queued events are flushed
  (appended to their series and propagated to downstream operators), and the
  new event starts a fresh queue.
* After all iterators are exhausted, any remaining queued events are flushed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import heapq
import time
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .operator import Operator
from .series import Array, Series
from .source import Source

type _AnySeries = Series[Any, Any]
type _AnySource = Source[Any, Any]
type _AnyOperator = Operator[Any, Any, Any, Any]
type _AnyArray = Array[Any, Any]
type _AnyEvent = tuple[np.datetime64, _SourceState, _AnyArray]


class Scenario:
    """A directed acyclic graph of sources and operators.

    Sources and operators are registered via :meth:`add_source` and
    :meth:`add_operator`, each returning the output :class:`~src.series.Series`
    that will be written to during :meth:`run`.
    """

    __slots__ = (
        "_sources",
        "_operators",
    )

    _sources: list[tuple[_AnySource, _AnySeries]]
    _operators: list[tuple[_AnyOperator, _AnySeries]]

    def __init__(self) -> None:
        self._sources = []
        self._operators = []

    @property
    def sources(self) -> list[tuple[_AnySource, _AnySeries]]:
        """Registered sources in insertion order."""
        return self._sources

    @property
    def operators(self) -> list[tuple[_AnyOperator, _AnySeries]]:
        """Registered operators in insertion order."""
        return self._operators

    def add_source(self, source: _AnySource) -> _AnySeries:
        """Register a source and return its output series."""
        series = Series(source.shape, source.dtype)
        self._sources.append((source, series))
        return series

    def add_operator(self, operator: _AnyOperator) -> _AnySeries:
        """Register an operator and return its output series."""
        series = Series(operator.shape, operator.dtype)
        self._operators.append((operator, series))
        return series

    async def run(self) -> None:
        """Consume all source streams and propagate to operators.

        Writes to the series returned by :meth:`add_source` and
        :meth:`add_operator`.
        """
        state = _ScenarioState(self)
        await state.run()


@dataclasses.dataclass(slots=True)
class _SourceState:
    """Per-source mutable state for one :meth:`Scenario.run` invocation.

    If both ``hist_task`` and ``pending_hist`` are ``None``, it means that
    the historical iterator is exhausted.  Similarly for the live iterator.
    """

    source: _AnySource
    series: _AnySeries
    hist_iter: AsyncIterator[tuple[np.datetime64, ArrayLike]]
    live_iter: AsyncIterator[ArrayLike]
    hist_task: asyncio.Task[Any] | None
    live_task: asyncio.Task[Any] | None
    pending_hist: tuple[np.datetime64, _AnyArray] | None
    pending_live: tuple[np.datetime64, _AnyArray] | None
    last_time: np.datetime64 | None  # last timestamp pushed into the POCQ


@dataclasses.dataclass(slots=True)
class _OperatorState:
    """Per-operator mutable state for one :meth:`Scenario.run` invocation."""

    operator: _AnyOperator
    series: _AnySeries
    state: Any


class _ScenarioState:
    """Per-run mutable state created at the start of :meth:`Scenario.run`.

    Owns operator computation states, frozen graph topology, and POCQ
    bookkeeping.
    """

    __slots__ = (
        "_edges",
        "_topo_order",
        "_sources",
        "_operators",
        "_queue",
        "_num_hist_tasks",
        "_pending_heap",
        "_heap_counter",
    )

    # Keyed by output series (always unique per node, even if same source/operator object
    # is registered multiple times).
    _edges: dict[_AnySeries, list[_AnySeries]]
    _topo_order: dict[_AnySeries, int]
    _sources: dict[_AnySeries, _SourceState]
    _operators: dict[_AnySeries, _OperatorState]
    _queue: list[_AnyEvent]
    _num_hist_tasks: int
    _pending_heap: list[tuple[np.datetime64, int, _SourceState, bool]]
    _heap_counter: int

    def __init__(self, scenario: Scenario) -> None:
        # Map: output series → list of downstream operator output series.
        self._edges = {}
        for _, series in scenario.sources:
            self._edges[series] = []
        for _, series in scenario.operators:
            self._edges[series] = []

        for operator, series in scenario.operators:
            for input in operator.inputs:
                if input not in self._edges:
                    raise ValueError("Operator input series must come from a registered source or operator output.")
                self._edges[input].append(series)

        # DFS topological sort with cycle detection.
        flags: dict[_AnySeries, bool] = {}
        exits: list[_AnySeries] = []

        def _dfs(s: _AnySeries) -> None:
            if s in flags:
                if flags[s]:
                    raise ValueError("Scenario operator dependency graph must be acyclic.")
                return
            flags[s] = True
            for child in self._edges[s]:
                _dfs(child)
            flags[s] = False
            exits.append(s)

        for _, series in scenario.operators:
            _dfs(series)

        topo = list(reversed(exits))
        self._topo_order = {s: i for i, s in enumerate(topo)}

        # Initialize source states (keyed by output series so the same source object
        # registered multiple times gets independent iterators and state).
        self._sources = {}
        for source, series in scenario.sources:
            hist_iter, live_iter = source.subscribe()
            st = _SourceState(
                source=source,
                series=series,
                hist_iter=hist_iter.__aiter__(),
                live_iter=live_iter.__aiter__(),
                hist_task=None,
                live_task=None,
                pending_hist=None,
                pending_live=None,
                last_time=None,
            )
            st.hist_task = asyncio.create_task(_anext(st.hist_iter))
            st.live_task = asyncio.create_task(_anext(st.live_iter))
            self._sources[series] = st

        # Initialize operator states (keyed by output series for the same reason).
        self._operators = {}
        for operator, series in scenario.operators:
            st = _OperatorState(
                operator,
                series,
                operator.init_state(),
            )
            self._operators[series] = st

        # Initialize POCQ.
        self._queue = []
        # O(1) check for whether all historical iterators have resolved.
        # Starts at len(sources) because every source begins with a hist_task.
        self._num_hist_tasks = len(scenario.sources)
        # Min-heap of pending events: (timestamp, tiebreaker, source_state, is_hist).
        # The tiebreaker avoids comparing _SourceState and ensures FIFO among
        # same-timestamp events.  Stale entries (where the corresponding
        # pending_hist/pending_live has already been consumed) are detected and
        # skipped on pop.
        self._pending_heap = []
        self._heap_counter = 0

    # -------------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Drives the POCQ until exhaustion."""

        try:
            while True:
                next_event = self._next_ready_event()
                if next_event is None:
                    if not await self._wait_for_events():
                        break
                else:
                    self._advance_queue(next_event)

            if self._queue:
                self._flush_queue()

        except Exception:
            tasks = [t for s in self._sources.values() for t in (s.hist_task, s.live_task) if t is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    # -------------------------------------------------------------------------
    # POCQ helpers
    # -------------------------------------------------------------------------

    def _next_ready_event(self) -> _AnyEvent | None:
        """Takes the pending event with minimum timestamp, or ``None``
        if blocked.

        The historical constraint requires every source with an active
        historical iterator to have a pending event before any timestamp
        can be committed.  Uses an O(1) counter instead of scanning all
        sources, and a min-heap for O(log N) minimum lookup.
        """
        if self._num_hist_tasks > 0:
            return None

        # Pop minimum from heap.
        while self._pending_heap:
            _, _, st, is_hist = heapq.heappop(self._pending_heap)
            pending = st.pending_hist if is_hist else st.pending_live
            assert pending is not None

            # Consume the event.
            time, value = pending
            st.last_time = time
            if is_hist:
                st.pending_hist = None
                st.hist_task = asyncio.create_task(_anext(st.hist_iter))
                self._num_hist_tasks += 1
            else:
                st.pending_live = None
                st.live_task = asyncio.create_task(_anext(st.live_iter))
            return time, st, value

        return None

    async def _wait_for_events(self) -> bool:
        """Wait for at least one source iterator to yield or exhaust."""

        tasks: dict[asyncio.Task[Any], tuple[_SourceState, bool]] = {}
        for st in self._sources.values():
            if st.hist_task is not None:
                tasks[st.hist_task] = (st, True)
            if st.live_task is not None:
                tasks[st.live_task] = (st, False)

        if not tasks:
            return False

        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            st, is_hist = tasks[task]

            if is_hist:
                st.hist_task = None
                self._num_hist_tasks -= 1
                try:
                    raw_time, raw_val = task.result()
                except StopAsyncIteration:
                    continue
                time = _coerce_timestamp(raw_time)
            else:
                st.live_task = None
                try:
                    raw_val = task.result()
                except StopAsyncIteration:
                    continue
                time = _runtime_timestamp()

            if st.last_time is not None and time < st.last_time:
                raise ValueError(
                    f"Source '{st.source.name}' emitted timestamp {time!r} which is less than "
                    f"last committed timestamp {st.last_time!r}."
                )

            value = np.asarray(raw_val, dtype=st.series.dtype)
            if value.shape != st.series.shape:
                raise ValueError(
                    f"Source '{st.source.name}' emitted value shape {value.shape}, " f"expected {st.series.shape}."
                )

            if is_hist:
                st.pending_hist = (time, value)
            else:
                st.pending_live = (time, value)

            self._heap_counter += 1
            heapq.heappush(self._pending_heap, (time, self._heap_counter, st, is_hist))

        return True

    def _advance_queue(self, next_event: _AnyEvent) -> None:
        """Push pending event into the POCQ, flushing first if needed."""
        time, _, _ = next_event
        if self._queue:
            prev_time, _, _ = self._queue[-1]
            if prev_time < time:
                self._flush_queue()

        self._queue.append(next_event)

    def _flush_queue(self) -> None:
        """Append POCQ events to their series and propagate to downstream operators.

        Downstream operators are processed in topological order via a min-heap
        keyed by each operator's topological order.
        """

        # Updated sources.
        updated: set[_AnySeries] = set()
        # Affected operators.
        affected: dict[_AnySeries, np.datetime64] = {}
        # Heap entries: (topo_order, series_id, op_series). Topo indices are
        # unique so ties never occur.
        heap: list[tuple[int, _AnySeries]] = []

        def _touch(child: _AnySeries, t: np.datetime64) -> None:
            if child in affected:
                affected[child] = max(affected[child], t)
            else:
                affected[child] = t
                heapq.heappush(heap, (self._topo_order[child], child))

        for time, st, value in reversed(self._queue):
            # When there are multiple events for the same source series, only the
            # most recent one (the last one in the queue) is relevant.
            if st.series in updated:
                continue
            updated.add(st.series)

            # POCQ guarantees that flushes have strictly increasing timestamps.
            st.series.append_unchecked(time, value)
            for child in self._edges[st.series]:
                _touch(child, time)

        while heap:
            _, op_series = heapq.heappop(heap)
            time = affected.pop(op_series)
            st = self._operators[op_series]
            raw_value, st.state = st.operator.compute(time, st.operator.inputs, st.state)

            # Returning None suppresses the output entry and halts propagation
            # to operators that are exclusively downstream of this one.
            if raw_value is None:
                continue

            value = np.asarray(raw_value, dtype=st.series.dtype)
            if value.shape != st.series.shape:
                raise ValueError(
                    f"Operator {type(st.operator).__name__!r} returned value shape "
                    f"{value.shape}, expected {st.series.shape}."
                )

            # POCQ guarantees that flushes have strictly increasing timestamps.
            st.series.append_unchecked(time, value)
            for child in self._edges[op_series]:
                _touch(child, time)

        self._queue.clear()


def _runtime_timestamp() -> np.datetime64:
    """Return the current wall-clock time as ``datetime64[ns]``."""
    return np.datetime64(time.time_ns(), "ns")


def _coerce_timestamp(value: np.datetime64) -> np.datetime64:
    """Coerce a timestamp-like value to ``datetime64[ns]``."""
    try:
        timestamp = np.datetime64(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not parse timestamp value {value!r}.") from exc
    return timestamp.astype("datetime64[ns]")


async def _anext[T](it: AsyncIterator[T]) -> T:
    """Wrapper for anext that produces a coroutine."""
    return await it.__anext__()
