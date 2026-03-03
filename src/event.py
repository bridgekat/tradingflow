"""Event model for timestamped source-series updates.

This module defines :class:`Event`, the runtime payload used by
:class:`~src.scenario.Scenario` to apply one logical update step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .series import Series


class Event:
    """A timestamped batch of source-series updates.

    An event contains:

    * a single ``np.datetime64`` timestamp;
    * a mapping from source :class:`~src.series.Series` objects to values that
    should be appended at that timestamp.

    Parameters
    ----------
    timestamp
        Event time represented as ``np.datetime64``.
    updates
        Optional mapping from source :class:`Series` objects to values.
        Values are consumed by :class:`~src.scenario.Scenario` and coerced
        to each source dtype at dispatch time.

    Invariants
    ----------
    * Event timestamps are represented as ``np.datetime64``.
    * Update keys are source-series objects (identity based), not names.
    * An event may contain zero updates.
    """

    __slots__ = ("_timestamp", "_updates")

    _timestamp: np.datetime64
    _updates: dict[Series[Any, Any], ArrayLike]

    def __init__(
        self,
        timestamp: np.datetime64,
        updates: Mapping[Series[Any, Any], ArrayLike] | None = None,
    ) -> None:
        if not isinstance(timestamp, np.datetime64):
            raise TypeError("Event timestamp must be a np.datetime64 scalar.")

        normalized_updates: dict[Series[Any, Any], ArrayLike] = {}
        if updates is not None:
            for series, value in updates.items():
                if not isinstance(series, Series):
                    raise TypeError("Event update keys must be Series instances.")
                normalized_updates[series] = value

        self._timestamp = timestamp
        self._updates = normalized_updates

    @property
    def timestamp(self) -> np.datetime64:
        """Timestamp of this event."""
        return self._timestamp

    @property
    def updates(self) -> Mapping[Series[Any, Any], ArrayLike]:
        """Source-series updates carried by this event."""
        return self._updates
