"""Scenario runtime -- thin Python wrapper around the Rust native backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from tradingflow._native import NativeArrayView, NativeSeriesView, NativeScenario
from . import Array, Series, operators
from .operator import Operator, NativeOperator
from .source import Source, NativeSource
from .types import Handle
from .views import ArrayView, SeriesView


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

        Shorthand for ``add_operator(Const(value))``.

        Parameters
        ----------
        value
            Initial value.
        """
        return self.add_operator(operators.Const(value))

    def add_source(self, source: Source | NativeSource) -> Handle:
        """Register a source and return a handle to its output node."""
        if isinstance(source, NativeSource):
            idx = self._native.add_native_source(
                source.kind,
                source.dtype,
                list(source.shape),
                source.params,
            )
            return Handle(idx, source.shape, np.dtype(source.dtype))
        else:
            idx = self._native.add_py_source(
                source,
                ("array", str(source.dtype)),
                list(source.shape),
            )
            return Handle(idx, source.shape, source.dtype)

    def add_operator(
        self,
        operator: NativeOperator | Operator,
        *,
        clock: Handle | None = None,
    ) -> Handle:
        """Register an operator and return a handle to its output node.

        Parameters
        ----------
        operator
            The operator to register (native or Python).
        clock
            Optional clock handle. If provided, the operator is triggered
            by the clock instead of its inputs. The clock is not an input —
            the operator does not read its value.
        """
        input_indices = [inp.index for inp in operator.inputs]
        if isinstance(operator, NativeOperator):
            idx = self._native.add_native_operator(
                operator.kind,
                str(operator.dtype),
                input_indices,
                list(operator.shape),
                operator.params,
                clock_index=clock.index if clock else None,
            )
        else:
            input_names, output_name = operator.get_io_types()
            idx = self._native.add_py_operator(
                input_indices,
                input_names,
                output_name,
                list(operator.shape),
                operator,
                operator.init_state(),
                clock_index=clock.index if clock else None,
            )
        return Handle(idx, operator.shape, operator.dtype)

    def run(self) -> None:
        """Execute the POCQ event loop.

        Python sources are driven by Rust-side async tasks that iterate
        the source's async iterators via a background asyncio event loop.
        The GIL is acquired briefly per event, preventing deadlocks with
        Python operators.
        """
        self._native.run()
