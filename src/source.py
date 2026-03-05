"""Base abstractions for source-series data ingestion.

This module defines the source-side protocol consumed by
:class:`src.scenario.Scenario`.

Public API
----------
Source[Shape, T]
    Abstract base class for one source bound to one source series.
SourceItem[Shape, T]
    One streamed data item carrying a value and optional payload timestamp.
TimestampMode
    Source timestamp semantics, either ``"payload"`` or ``"ingest"``.

Invariants
----------
* One source owns exactly one source series.
* Source-level timestamp monotonicity is enforced strictly.
* Values are validated against the bound series shape and dtype before append.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike

from .series import AnyShape, Array, Series

type TimestampMode = Literal["payload", "ingest"]


@dataclass(slots=True, frozen=True)
class SourceItem[Shape: AnyShape, T: np.generic]:
    """One streamed data item.

    Parameters
    ----------
    value
        The value to append to the bound source series.
    timestamp
        Optional payload timestamp. Required for payload-timestamp sources;
        ignored for ingest-timestamp sources.
    """

    value: ArrayLike
    timestamp: np.datetime64 | None = None


class Source[Shape: AnyShape, T: np.generic](ABC):
    """Abstract base class for one source bound to one source series.

    Subclasses implement :meth:`stream` and yield :class:`SourceItem`
    instances. The runtime validates value dtype/shape and timestamp
    monotonicity before appending into :attr:`series`.
    """

    __slots__ = ("_series", "_name", "_timestamp_mode", "_last_timestamp")

    _series: Series[Shape, T]
    _name: str
    _timestamp_mode: TimestampMode
    _last_timestamp: np.datetime64 | None

    def __init__(
        self,
        series: Series[Shape, T],
        *,
        name: str | None = None,
        timestamp_mode: TimestampMode,
    ) -> None:
        self._series = series
        self._name = name or type(self).__name__
        self._timestamp_mode = timestamp_mode
        self._last_timestamp = series.index[-1] if len(series) > 0 else None

    @abstractmethod
    async def stream(self) -> AsyncIterator[SourceItem[Shape, T]]:
        """Returns an async stream of source items."""
        raise NotImplementedError

    @property
    def series(self) -> Series[Shape, T]:
        """The source series owned by this source."""
        return self._series

    @property
    def name(self) -> str:
        """Human-readable source name."""
        return self._name

    @property
    def timestamp_mode(self) -> TimestampMode:
        """Timestamp semantics of this source."""
        return self._timestamp_mode

    @property
    def last_timestamp(self) -> np.datetime64 | None:
        """Last committed timestamp emitted by this source."""
        return self._last_timestamp

    def normalize_item(
        self,
        item: SourceItem[Shape, T],
        *,
        ingest_timestamp: np.datetime64 | None = None,
    ) -> tuple[np.datetime64, Array[Shape, T]]:
        """Validates and normalizes one streamed item."""
        timestamp = self._resolve_timestamp(item.timestamp, ingest_timestamp=ingest_timestamp)
        value = np.asarray(item.value, dtype=self._series.dtype)
        if value.shape != self._series.shape:
            raise ValueError(
                f"Source '{self._name}' emitted value shape {value.shape}, " f"expected {self._series.shape}"
            )
        return timestamp, cast(Array[Shape, T], value)

    def commit_timestamp(self, timestamp: np.datetime64) -> None:
        """Commits a timestamp after successful append."""
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                f"Source '{self._name}' emitted non-increasing timestamp {timestamp!r}; "
                f"last timestamp is {self._last_timestamp!r}"
            )
        self._last_timestamp = timestamp

    def _resolve_timestamp(
        self,
        payload_timestamp: np.datetime64 | None,
        *,
        ingest_timestamp: np.datetime64 | None,
    ) -> np.datetime64:
        if self._timestamp_mode == "payload":
            if payload_timestamp is None:
                raise ValueError(f"Source '{self._name}' requires payload timestamps.")
            timestamp = _to_datetime64_ns(payload_timestamp)
        else:
            if ingest_timestamp is None:
                raise ValueError(f"Source '{self._name}' requires ingest timestamps from the runtime.")
            timestamp = _to_datetime64_ns(ingest_timestamp)
            if self._last_timestamp is not None and timestamp <= self._last_timestamp:
                timestamp = cast(np.datetime64, self._last_timestamp + np.timedelta64(1, "ns"))

        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError(
                f"Source '{self._name}' emitted non-increasing timestamp {timestamp!r}; "
                f"last timestamp is {self._last_timestamp!r}"
            )
        return timestamp


def _to_datetime64_ns(value: np.datetime64 | Any) -> np.datetime64:
    """Coerces a timestamp-like value to ``datetime64[ns]``."""
    try:
        timestamp = np.datetime64(value)
    except Exception as exc:  # pragma: no cover - exact NumPy error type varies.
        raise ValueError(f"Could not parse timestamp value {value!r}.") from exc
    return cast(np.datetime64, timestamp.astype("datetime64[ns]"))
