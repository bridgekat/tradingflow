"""Source interface for data feeding into the computation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .data.types import NodeKind, _to_native_node_kind

if TYPE_CHECKING:
    from tradingflow._native import NativeScenario


class Source(ABC):
    """Abstract base for data sources.

    Parameters
    ----------
    dtype
        NumPy dtype for emitted values (ignored for unit sources).
    shape
        Shape of each emitted value element. Use `()` for scalars.
    kind
        Output node kind (default [`NodeKind.ARRAY`]).
    initial
        Initial value. Defaults to NaN for floats, zero otherwise.
    name
        Optional human-readable name.
    """

    __slots__ = ("_dtype", "_shape", "_kind", "_initial", "_name")

    def __init__(
        self,
        dtype: type | np.dtype,
        shape: tuple[int, ...],
        *,
        kind: NodeKind = NodeKind.ARRAY,
        initial: ArrayLike | None = None,
        name: str | None = None,
    ) -> None:
        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._kind = kind
        if initial is not None:
            self._initial = np.asarray(initial, dtype=self._dtype)
        elif np.issubdtype(self._dtype, np.floating):
            self._initial = np.full(shape, np.nan, dtype=self._dtype)
        else:
            self._initial = np.zeros(shape, dtype=self._dtype)
        self._name = name or type(self).__name__

    @abstractmethod
    def init(self, timestamp: int) -> tuple[
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
        AsyncIterator[tuple[np.datetime64, ArrayLike]],
    ]:
        """Return a `(historical, live)` async-iterator pair.

        Both iterators yield `(timestamp, value)` tuples.  Timestamps
        are `int64` TAI nanoseconds since the PTP epoch (1970-01-01
        00:00:00 TAI) — the same convention NumPy `datetime64[ns]` uses
        numerically.  The bridge reinterprets them directly; no
        leap-second math runs here.

        Parameters
        ----------
        timestamp
            Initial timestamp (TAI nanoseconds since the PTP epoch).
        """
        ...

    @property
    def kind(self) -> NodeKind:
        """Output node kind."""
        return self._kind

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype for emitted values."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape of each emitted value."""
        return self._shape

    @property
    def initial(self) -> np.ndarray:
        """Initial value array."""
        return self._initial

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name

    def _register(self, native_scenario: NativeScenario) -> int:
        """Register this Python source with the native scenario.

        Polymorphic dispatch: [`Scenario.add_source`][tradingflow.Scenario.add_source]
        delegates to this method without branching on source kind.
        """
        return native_scenario.add_py_source(
            self,
            _to_native_node_kind(self._kind),
            "" if self._kind == NodeKind.UNIT else self._dtype.name,
            list(self._shape),
        )


async def empty_historical_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting historical async generator."""
    return
    yield


async def empty_live_gen() -> AsyncIterator[tuple[np.datetime64, Any]]:
    """Immediately-exhausting live async generator."""
    return
    yield


class NativeSource:
    """Descriptor for a Rust-implemented source.

    Analogous to [`NativeOperator`][tradingflow.NativeOperator] -- carries
    `native_id` + `params` and is dispatched entirely on the native side. Not a
    [`Source`][tradingflow.Source] subclass (no Python async iterators).

    Parameters
    ----------
    native_id
        Source native dispatch string on the Rust side.
    kind
        Output node kind (default [`NodeKind.ARRAY`]).
    dtype
        NumPy dtype string (default `"float64"`; ignored for `NodeKind.UNIT`).
    shape
        Element shape (default `()`).
    params
        Source-specific parameters.
    name
        Optional human-readable name (defaults to *native_id*).
    """

    __slots__ = ("_native_id", "_kind", "_dtype", "_shape", "_params", "_name")

    def __init__(
        self,
        native_id: str,
        *,
        kind: NodeKind = NodeKind.ARRAY,
        dtype: str = "float64",
        shape: tuple[int, ...] = (),
        params: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._native_id = native_id
        self._kind = kind
        self._dtype = dtype
        self._shape = shape
        self._params = params or {}
        self._name = name or native_id

    @property
    def native_id(self) -> str:
        """Source native dispatch string."""
        return self._native_id

    @property
    def kind(self) -> NodeKind:
        """Output node kind."""
        return self._kind

    @property
    def dtype(self) -> str:
        """NumPy dtype string."""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Element shape."""
        return self._shape

    @property
    def params(self) -> dict:
        """Source-specific parameters."""
        return self._params

    @property
    def name(self) -> str:
        """Human-readable name."""
        return self._name

    def _register(self, native_scenario: NativeScenario) -> int:
        """Register this native source with the native scenario."""
        return native_scenario.add_native_source(
            self._native_id,
            self._dtype,
            list(self._shape),
            self._params,
        )
