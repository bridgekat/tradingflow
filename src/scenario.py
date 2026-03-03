"""Scenario runtime for event-driven series/operator execution.

This module defines :class:`Scenario`, which owns a directed acyclic graph
of source series and derived-series operators.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from .event import Event
from .operator import Operator
from .series import Series


type _AnySeries = Series[Any, Any]
type _AnyOperator = Operator[Any, Any, Any, Any]


class Scenario:
    """Event-driven execution context for source and derived series.

    Key responsibilities:

    * Register source :class:`~src.series.Series` objects and
    :class:`~src.operator.Operator` objects.
    * Freeze and validate dependencies on first dispatch.
    * Apply :class:`~src.event.Event` updates to source series.
    * Propagate updates only to affected downstream operators in topological order.

    Parameters
    ----------
    sources
        Source series that can be updated directly by events.
    operators
        Operators that derive series from sources and/or other operators.

    Invariants
    ----------
    * Event timestamps dispatched to a scenario are strictly increasing.
    * The operator dependency graph must be acyclic.
    * After the first dispatch, graph structure is frozen.
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

    _sources: set[_AnySeries]
    _operators: list[_AnyOperator]
    _frozen: bool
    _last_timestamp: np.datetime64 | None
    _topological_operators: tuple[_AnyOperator, ...]
    _source_consumers: dict[_AnySeries, tuple[_AnyOperator, ...]]
    _downstream: dict[_AnyOperator, tuple[_AnyOperator, ...]]

    def __init__(
        self,
        sources: Iterable[_AnySeries] = (),
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

    def add_source(self, source: _AnySeries) -> None:
        """Registers a source series before the scenario is frozen."""
        if self._frozen:
            raise RuntimeError("Scenario is frozen and no longer accepts new sources.")
        if not isinstance(source, Series):
            raise TypeError("source must be a Series instance.")
        self._sources.add(source)

    def add_operator(self, operator: _AnyOperator) -> None:
        """Registers an operator before the scenario is frozen."""
        if self._frozen:
            raise RuntimeError("Scenario is frozen and no longer accepts new operators.")
        if not isinstance(operator, Operator):
            raise TypeError("operator must be an Operator instance.")
        if operator not in self._operators:
            self._operators.append(operator)

    def dispatch(self, event: Event) -> None:
        """Applies an event and updates all affected downstream operators."""
        if not isinstance(event, Event):
            raise TypeError("event must be an Event instance.")

        if not self._frozen:
            self._freeze_graph()

        if self._last_timestamp is not None and event.timestamp <= self._last_timestamp:
            raise ValueError(
                f"event timestamp {event.timestamp!r} is not greater than the last dispatched "
                f"timestamp {self._last_timestamp!r}"
            )

        affected: set[_AnyOperator] = set()
        for source, value in event.updates.items():
            if source not in self._sources:
                raise ValueError("Event contains an update for an unregistered source series.")

            array = np.asarray(value, dtype=source.dtype)
            if array.shape != source.shape:
                raise ValueError(f"update shape {array.shape} does not match source shape {source.shape}")

            source.append(event.timestamp, cast(Any, array))
            affected.update(self._source_consumers.get(source, ()))

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
                    operator.update(event.timestamp)

        self._last_timestamp = event.timestamp

    def dispatch_many(self, events: Iterable[Event]) -> None:
        """Dispatches multiple events in sequence."""
        for event in events:
            self.dispatch(event)

    def _freeze_graph(self) -> None:
        """Builds dependency structures and validates acyclicity."""
        outputs_to_producers: dict[_AnySeries, _AnyOperator] = {}
        for operator in self._operators:
            outputs_to_producers[cast(_AnySeries, operator.output)] = operator

        source_consumers_sets: dict[_AnySeries, set[_AnyOperator]] = {source: set() for source in self._sources}
        downstream_sets: dict[_AnyOperator, set[_AnyOperator]] = {operator: set() for operator in self._operators}
        indegree: dict[_AnyOperator, int] = {operator: 0 for operator in self._operators}
        registration_index = {operator: i for i, operator in enumerate(self._operators)}

        for consumer in self._operators:
            for input_series in consumer.inputs:
                input_series_any = cast(_AnySeries, input_series)
                if input_series_any in self._sources:
                    source_consumers_sets[input_series_any].add(consumer)
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
