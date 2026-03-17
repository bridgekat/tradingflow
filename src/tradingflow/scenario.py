"""Scenario runtime for source-driven observable/operator execution.

This module defines [`Scenario`][tradingflow.Scenario], a specification of sources and
operators that form a directed acyclic graph of observable values, and
[`_ScenarioState`][tradingflow.scenario._ScenarioState], the per-run mutable state created fresh for
every [`Scenario.run`][tradingflow.Scenario.run] invocation.

Point-of-coherency queue (POCQ)
-------------------------------
Each source exposes a `(historical, live)` iterator pair via
[`Source.subscribe`][tradingflow.Source.subscribe].  The runtime converts all incoming items
into `(timestamp, source_state, value)` events and accumulates them in the POCQ.

* **Historical constraint** – before advancing the POCQ, every active
  historical iterator must have a pending event ready.  This prevents an
  in-flight historical iterator from later producing a timestamp smaller than
  one already committed.
* **Live iterators** are exempt from this constraint: their timestamps are
  wall-clock instants and are therefore always >= any historical timestamp
  and any existing live timestamp.
* The POCQ accumulates events sharing the same timestamp.  When an event with
  a strictly larger timestamp arrives, all queued events are flushed
  (written to their observables, copied to materialized series, and propagated
  to downstream operators), and the new event starts a fresh queue.
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

from .observable import Observable
from .operator import Operator
from .series import Array, Series
from .source import Source

type _AnyObservable = Observable[Any, Any]
type _AnySeries = Series[Any, Any]
type _AnySource = Source[Any, Any]
type _AnyOperator = Operator[Any, Any, Any, Any]
type _AnyArray = Array[Any, Any]
type _AnyEvent = tuple[np.datetime64, _SourceState, _AnyArray]


class Scenario:
    """A directed acyclic graph of sources and operators.

    Sources and operators are registered via [`add_source`][.add_source] and
    [`add_operator`][.add_operator], each returning the output
    [`Observable`][tradingflow.Observable] that will be updated during
    [`run`][.run].  Observables can be *materialized* into
    [`Series`][tradingflow.Series] via [`materialize`][.materialize].
    """

    __slots__ = (
        "_sources",
        "_operators",
        "_materializations",
    )

    _sources: list[tuple[_AnySource, _AnyObservable]]
    _operators: list[tuple[_AnyOperator, _AnyObservable]]
    _materializations: dict[int, _AnySeries]  # id(observable) -> series

    def __init__(self) -> None:
        self._sources = []
        self._operators = []
        self._materializations = {}

    @property
    def sources(self) -> list[tuple[_AnySource, _AnyObservable]]:
        """Registered sources in insertion order."""
        return self._sources

    @property
    def operators(self) -> list[tuple[_AnyOperator, _AnyObservable]]:
        """Registered operators in insertion order."""
        return self._operators

    def add_source(self, source: _AnySource) -> _AnyObservable:
        """Register a source and return its output observable."""
        observable = Observable(source.shape, source.dtype)
        self._sources.append((source, observable))
        return observable

    def add_operator(self, operator: _AnyOperator) -> _AnyObservable:
        """Register an operator and return its output observable."""
        observable = Observable(operator.shape, operator.dtype)
        self._operators.append((operator, observable))
        return observable

    def materialize(self, observable: _AnyObservable) -> _AnySeries:
        """Materialize an observable: allocate a series to store full history.

        Returns the [`Series`][tradingflow.Series] that will be populated
        during [`run`][.run].
        """
        key = id(observable)
        if key in self._materializations:
            return self._materializations[key]
        series = Series(observable.shape, observable.dtype)
        self._materializations[key] = series
        return series

    async def run(self) -> None:
        """Consume all source streams and propagate to operators.

        Updates the observables returned by [`add_source`][.add_source] and
        [`add_operator`][.add_operator], and appends to materialized series.
        """
        state = _ScenarioState(self)
        await state.run()


@dataclasses.dataclass(slots=True)
class _SourceState:
    """Per-source mutable state for one [`Scenario.run`][tradingflow.Scenario.run] invocation."""

    source: _AnySource
    observable: _AnyObservable
    series: _AnySeries | None
    hist_iter: AsyncIterator[tuple[np.datetime64, ArrayLike]]
    live_iter: AsyncIterator[ArrayLike]
    hist_task: asyncio.Task[Any] | None
    live_task: asyncio.Task[Any] | None
    pending_hist: tuple[np.datetime64, _AnyArray] | None
    pending_live: tuple[np.datetime64, _AnyArray] | None
    last_time: np.datetime64 | None


@dataclasses.dataclass(slots=True)
class _OperatorState:
    """Per-operator mutable state for one [`Scenario.run`][tradingflow.Scenario.run] invocation."""

    operator: _AnyOperator
    observable: _AnyObservable
    series: _AnySeries | None
    state: Any


class _ScenarioState:
    """Per-run mutable state created at the start of [`Scenario.run`][tradingflow.Scenario.run].

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

    _edges: dict[int, list[int]]  # id(observable) -> [id(downstream_observable)]
    _topo_order: dict[int, int]  # id(observable) -> topological rank
    _sources: dict[int, _SourceState]  # id(observable) -> state
    _operators: dict[int, _OperatorState]  # id(observable) -> state
    _queue: list[_AnyEvent]
    _num_hist_tasks: int
    _pending_heap: list[tuple[np.datetime64, int, _SourceState, bool]]
    _heap_counter: int

    def __init__(self, scenario: Scenario) -> None:
        # Collect all observables (sources + operators).
        all_obs: list[_AnyObservable] = []
        for _, obs in scenario.sources:
            all_obs.append(obs)
        for _, obs in scenario.operators:
            all_obs.append(obs)

        # Map: id(observable) -> list of downstream operator id(observable).
        self._edges = {id(obs): [] for obs in all_obs}

        for operator, obs in scenario.operators:
            for inp in operator.inputs:
                # An input can be an Observable or a Series.
                # Find the id of the input in our node set.
                inp_id = self._resolve_input_id(inp, scenario)
                if inp_id not in self._edges:
                    raise ValueError("Operator input must come from a registered source or operator output.")
                self._edges[inp_id].append(id(obs))

        # DFS topological sort with cycle detection.
        flags: dict[int, bool] = {}
        exits: list[int] = []

        def _dfs(node_id: int) -> None:
            if node_id in flags:
                if flags[node_id]:
                    raise ValueError("Scenario operator dependency graph must be acyclic.")
                return
            flags[node_id] = True
            for child_id in self._edges[node_id]:
                _dfs(child_id)
            flags[node_id] = False
            exits.append(node_id)

        for _, obs in scenario.operators:
            _dfs(id(obs))

        topo = list(reversed(exits))
        self._topo_order = {node_id: i for i, node_id in enumerate(topo)}

        # Initialize source states.
        self._sources = {}
        for source, obs in scenario.sources:
            hist_iter, live_iter = source.subscribe()
            st = _SourceState(
                source=source,
                observable=obs,
                series=scenario._materializations.get(id(obs)),
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
            self._sources[id(obs)] = st

        # Initialize operator states.
        self._operators = {}
        for operator, obs in scenario.operators:
            st = _OperatorState(
                operator=operator,
                observable=obs,
                series=scenario._materializations.get(id(obs)),
                state=operator.init_state(),
            )
            self._operators[id(obs)] = st

        # Initialize observable values.
        # 1. Write source initial values to their observables.
        for source, obs in scenario.sources:
            obs.write(source.initial)
        # 2. Run operators once in topological order to compute initial outputs.
        for node_id in topo:
            if node_id in self._operators:
                st = self._operators[node_id]
                raw_value, st.state = st.operator.compute(
                    np.datetime64(0, "ns"), st.operator.inputs, st.state
                )
                if raw_value is not None:
                    st.observable.write(np.asarray(raw_value, dtype=st.observable.dtype))

        # Initialize POCQ.
        self._queue = []
        self._num_hist_tasks = len(scenario.sources)
        self._pending_heap = []
        self._heap_counter = 0

    @staticmethod
    def _resolve_input_id(inp: Any, scenario: Scenario) -> int:
        """Resolve an operator input (Observable or materialized Series) to a node id."""
        # Direct observable reference.
        if isinstance(inp, Observable):
            return id(inp)
        # Materialized series — find the observable it was materialized from.
        if isinstance(inp, Series):
            for obs_id, series in scenario._materializations.items():
                if series is inp:
                    return obs_id
            raise ValueError("Series input must be a materialized series from this scenario.")
        raise TypeError(f"Unsupported input type: {type(inp)}")

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
        """Takes the pending event with minimum timestamp, or `None`
        if blocked.
        """
        if self._num_hist_tasks > 0:
            return None

        while self._pending_heap:
            _, _, st, is_hist = heapq.heappop(self._pending_heap)
            pending = st.pending_hist if is_hist else st.pending_live
            assert pending is not None

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

            value = np.asarray(raw_val, dtype=st.observable.dtype)
            if value.shape != st.observable.shape:
                raise ValueError(
                    f"Source '{st.source.name}' emitted value shape {value.shape}, "
                    f"expected {st.observable.shape}."
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
        """Write POCQ events to observables and propagate to downstream operators.

        For each updated node, the observable is written first.  If the node
        has a materialized series, the value is also appended to it.
        Downstream operators are processed in topological order via a min-heap.
        """

        # Updated source observables.
        updated: set[int] = set()
        # Affected operator observables.
        affected: dict[int, np.datetime64] = {}
        heap: list[tuple[int, int]] = []

        def _touch(child_id: int, t: np.datetime64) -> None:
            if child_id in affected:
                affected[child_id] = max(affected[child_id], t)
            else:
                affected[child_id] = t
                heapq.heappush(heap, (self._topo_order[child_id], child_id))

        for time, st, value in reversed(self._queue):
            obs_id = id(st.observable)
            if obs_id in updated:
                continue
            updated.add(obs_id)

            # Update observable.
            st.observable.write(value)
            # Materialize to series if applicable.
            if st.series is not None:
                st.series.append_unchecked(time, value)

            for child_id in self._edges[obs_id]:
                _touch(child_id, time)

        while heap:
            _, op_obs_id = heapq.heappop(heap)
            time = affected.pop(op_obs_id)
            st = self._operators[op_obs_id]
            raw_value, st.state = st.operator.compute(time, st.operator.inputs, st.state)

            if raw_value is None:
                continue

            value = np.asarray(raw_value, dtype=st.observable.dtype)
            if value.shape != st.observable.shape:
                raise ValueError(
                    f"Operator {type(st.operator).__name__!r} returned value shape "
                    f"{value.shape}, expected {st.observable.shape}."
                )

            # Update observable.
            st.observable.write(value)
            # Materialize to series if applicable.
            if st.series is not None:
                st.series.append_unchecked(time, value)

            for child_id in self._edges[op_obs_id]:
                _touch(child_id, time)

        self._queue.clear()


def _runtime_timestamp() -> np.datetime64:
    """Return the current wall-clock time as `datetime64[ns]`."""
    return np.datetime64(time.time_ns(), "ns")


def _coerce_timestamp(value: np.datetime64) -> np.datetime64:
    """Coerce a timestamp-like value to `datetime64[ns]`."""
    try:
        timestamp = np.datetime64(value)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Could not parse timestamp value {value!r}.") from exc
    return timestamp.astype("datetime64[ns]")


async def _anext[T](it: AsyncIterator[T]) -> T:
    """Wrapper for anext that produces a coroutine."""
    return await it.__anext__()
