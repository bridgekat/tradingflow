"""Scenario runtime for source-driven series/operator execution.

This module defines :class:`Scenario`, which owns a directed acyclic graph
of source series and derived-series operators, and drives the graph from
asynchronous source streams.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

import numpy as np

from .operator import Operator
from .series import Array, Series
from .sources import Source, SourceItem


type _AnySeries = Series[Any, Any]
type _AnySource = Source[Any, Any]
type _AnyOperator = Operator[Any, Any, Any, Any]
type _AnyArray = Array[Any, Any]


class Scenario:
    """Async execution context for source and derived series.

    Key responsibilities:

    * Register source :class:`~src.sources.Source` objects and
      :class:`~src.operator.Operator` objects.
    * Freeze and validate dependencies before the first run.
    * Read source streams and append updates into source series.
    * Coalesce same-timestamp updates and propagate them to affected
      downstream operators in topological order.

    Parameters
    ----------
    sources
        Source objects that can append to source series.
    operators
        Operators that derive series from sources and/or other operators.

    Invariants
    ----------
    * Each source owns exactly one source series.
    * The operator dependency graph must be acyclic.
    * After the first run starts, graph structure is frozen.
    * Committed runtime timestamps are strictly increasing.
    """

    __slots__ = (
        "_sources",
        "_operators",
        "_frozen",
        "_last_timestamp",
        "_topological_operators",
        "_source_consumers",
        "_downstream",
    )

    _sources: set[_AnySource]
    _operators: list[_AnyOperator]
    _frozen: bool
    _last_timestamp: np.datetime64 | None
    _topological_operators: tuple[_AnyOperator, ...]
    _source_consumers: dict[_AnySource, tuple[_AnyOperator, ...]]
    _downstream: dict[_AnyOperator, tuple[_AnyOperator, ...]]

    def __init__(
        self,
        sources: Iterable[_AnySource] = (),
        operators: Iterable[_AnyOperator] = (),
    ) -> None:
        self._sources = set()
        self._operators = []
        self._frozen = False
        self._last_timestamp = None
        self._topological_operators = ()
        self._source_consumers = {}
        self._downstream = {}

        for source in sources:
            self.add_source(source)
        for operator in operators:
            self.add_operator(operator)

    def add_source(self, source: _AnySource) -> None:
        """Registers a source before the scenario is frozen."""
        if self._frozen:
            raise RuntimeError("Scenario is frozen and no longer accepts new sources.")
        if not isinstance(source, Source):
            raise TypeError("source must be a Source instance.")
        if source in self._sources:
            return
        for existing in self._sources:
            if existing.series is source.series:
                raise ValueError("Two sources cannot own the same source series.")
        self._sources.add(source)

    def add_operator(self, operator: _AnyOperator) -> None:
        """Registers an operator before the scenario is frozen."""
        if self._frozen:
            raise RuntimeError("Scenario is frozen and no longer accepts new operators.")
        if not isinstance(operator, Operator):
            raise TypeError("operator must be an Operator instance.")
        if operator not in self._operators:
            self._operators.append(operator)

    async def run(self) -> None:
        """Consumes all source streams and updates affected operators."""
        if not self._frozen:
            self._freeze_graph()

        iterators: dict[_AnySource, AsyncIterator[SourceItem[Any, Any]]] = {}
        active_sources: set[_AnySource] = set(self._sources)
        tasks: dict[_AnySource, asyncio.Task[SourceItem[Any, Any]]] = {}
        pending: dict[_AnySource, tuple[np.datetime64, _AnyArray]] = {}

        for source in self._sources:
            iterator = source.stream()
            iterators[source] = iterator.__aiter__()
            tasks[source] = asyncio.create_task(anext(iterators[source]))

        try:
            while active_sources or pending:
                timestamp = self._next_ready_timestamp(pending, active_sources)
                if timestamp is None:
                    if not tasks:
                        break
                    await self._collect_next_items(tasks, pending, active_sources)
                    continue
                self._apply_timestamp_batch(timestamp, pending, tasks, iterators, active_sources)
        except BaseException:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            raise

    async def _collect_next_items(
        self,
        tasks: dict[_AnySource, asyncio.Task[SourceItem[Any, Any]]],
        pending: dict[_AnySource, tuple[np.datetime64, _AnyArray]],
        active_sources: set[_AnySource],
    ) -> None:
        """Waits until at least one source yields or exhausts."""
        if not tasks:
            return

        done, _ = await asyncio.wait(tuple(tasks.values()), return_when=asyncio.FIRST_COMPLETED)
        source_by_task = {task: source for source, task in tasks.items()}

        for task in done:
            source = source_by_task[task]
            del tasks[source]
            try:
                item = task.result()
            except StopAsyncIteration:
                active_sources.discard(source)
                continue

            ingest_timestamp = self._runtime_timestamp() if source.timestamp_mode == "ingest" else None
            timestamp, value = source.normalize_item(item, ingest_timestamp=ingest_timestamp)
            pending[source] = (timestamp, cast(_AnyArray, value))

    def _next_ready_timestamp(
        self,
        pending: dict[_AnySource, tuple[np.datetime64, _AnyArray]],
        active_sources: set[_AnySource],
    ) -> np.datetime64 | None:
        """Returns the next timestamp eligible for commit, if any."""
        if not pending:
            return None

        active_payload = {source for source in active_sources if source.timestamp_mode == "payload"}
        if active_payload and not active_payload.issubset(pending):
            return None

        return min(timestamp for timestamp, _ in pending.values())

    def _apply_timestamp_batch(
        self,
        timestamp: np.datetime64,
        pending: dict[_AnySource, tuple[np.datetime64, _AnyArray]],
        tasks: dict[_AnySource, asyncio.Task[SourceItem[Any, Any]]],
        iterators: dict[_AnySource, AsyncIterator[SourceItem[Any, Any]]],
        active_sources: set[_AnySource],
    ) -> None:
        """Appends all pending source updates at one timestamp and propagates."""
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                f"Scenario received non-increasing timestamp {timestamp!r}; "
                f"last committed timestamp is {self._last_timestamp!r}."
            )

        affected: set[_AnyOperator] = set()
        consumed_sources: list[_AnySource] = []

        for source, (source_timestamp, value) in tuple(pending.items()):
            if source_timestamp != timestamp:
                continue
            source.series.append(timestamp, value)
            source.commit_timestamp(timestamp)
            affected.update(self._source_consumers.get(source, ()))
            consumed_sources.append(source)
            del pending[source]

        for source in consumed_sources:
            if source in active_sources:
                tasks[source] = asyncio.create_task(anext(iterators[source]))

        if affected:
            stack = list(affected)
            while stack:
                operator = stack.pop()
                for child in self._downstream.get(operator, ()):
                    if child in affected:
                        continue
                    affected.add(child)
                    stack.append(child)

            for operator in self._topological_operators:
                if operator in affected:
                    operator.update(timestamp)

        self._last_timestamp = timestamp

    def _freeze_graph(self) -> None:
        """Builds dependency structures and validates acyclicity."""
        outputs_to_producers: dict[_AnySeries, _AnyOperator] = {}
        for operator in self._operators:
            outputs_to_producers[cast(_AnySeries, operator.output)] = operator

        source_by_series: dict[_AnySeries, _AnySource] = {}
        for source in self._sources:
            series_any = cast(_AnySeries, source.series)
            if series_any in source_by_series:
                raise ValueError("Two sources cannot own the same source series.")
            source_by_series[series_any] = source

        source_consumers_sets: dict[_AnySource, set[_AnyOperator]] = {source: set() for source in self._sources}
        downstream_sets: dict[_AnyOperator, set[_AnyOperator]] = {operator: set() for operator in self._operators}
        indegree: dict[_AnyOperator, int] = {operator: 0 for operator in self._operators}
        registration_index = {operator: i for i, operator in enumerate(self._operators)}

        for consumer in self._operators:
            for input_series in consumer.inputs:
                input_series_any = cast(_AnySeries, input_series)
                source = source_by_series.get(input_series_any)
                if source is not None:
                    source_consumers_sets[source].add(consumer)
                    continue

                producer = outputs_to_producers.get(input_series_any)
                if producer is None:
                    raise ValueError("Operator input series must come from a registered source or operator output.")

                if consumer not in downstream_sets[producer]:
                    downstream_sets[producer].add(consumer)
                    indegree[consumer] += 1

        zero_indegree = deque(operator for operator in self._operators if indegree[operator] == 0)
        topological_order: list[_AnyOperator] = []
        while zero_indegree:
            operator = zero_indegree.popleft()
            topological_order.append(operator)

            children = sorted(downstream_sets[operator], key=registration_index.__getitem__)
            for child in children:
                indegree[child] -= 1
                if indegree[child] == 0:
                    zero_indegree.append(child)

        if len(topological_order) != len(self._operators):
            raise ValueError("Scenario operator dependency graph must be acyclic.")

        self._topological_operators = tuple(topological_order)
        self._source_consumers = {
            source: tuple(sorted(consumers, key=registration_index.__getitem__))
            for source, consumers in source_consumers_sets.items()
        }
        self._downstream = {
            operator: tuple(sorted(children, key=registration_index.__getitem__))
            for operator, children in downstream_sets.items()
        }
        self._frozen = True

    @staticmethod
    def _runtime_timestamp() -> np.datetime64:
        """Returns the current runtime timestamp in nanoseconds."""
        return np.datetime64(time.time_ns(), "ns")
