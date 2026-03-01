"""Rolling predictor base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series
from ...series import T


class RollingPredictor(Operator[dict[str, Any], T], ABC):
    """Abstract base for prediction models with rolling retraining."""

    __slots__ = ("_train_window", "_retrain_every", "_update_count")

    def __init__(
        self,
        train_window: int,
        retrain_every: int,
        inputs: list[Series[Any]],
        state: dict[str, Any],
        dtype: np.dtype[T],
        shape: tuple[int, ...] = (),
    ) -> None:
        super().__init__(inputs, state, dtype, shape)
        self._train_window = train_window
        self._retrain_every = retrain_every
        self._update_count = 0

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        if not all(inputs):
            return None
        self._update_count += 1
        if self._update_count == 1 or self._update_count % self._retrain_every == 0:
            self._fit(timestamp, *inputs)
        return self._predict(timestamp, *inputs)

    @abstractmethod
    def _fit(self, timestamp: np.datetime64, *inputs: Series[Any]) -> None:
        """Retrain the model on the current rolling window of data."""

    @abstractmethod
    def _predict(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        """Produce a prediction using the currently fitted model."""

    @property
    def train_window(self) -> int:
        """Number of entries used for training."""
        return self._train_window

    @property
    def retrain_every(self) -> int:
        """Retraining frequency (in number of updates)."""
        return self._retrain_every
