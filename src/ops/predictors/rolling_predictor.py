"""Abstract rolling predictor base class.

Provides :class:`RollingPredictor`, an :class:`Operator` subclass that
automatically retrains a model every *retrain_every* steps on a window
of the most recent *train_window* observations.  Subclasses implement
:meth:`_fit` (training) and :meth:`_predict` (inference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class RollingPredictor(Operator, ABC):
    """Abstract base for predictors that retrain on a rolling window.

    Subclasses must implement :meth:`_fit` and :meth:`_predict`.
    The predictor retrains every *retrain_every* calls to :meth:`update`.
    """

    __slots__ = ("_train_window", "_retrain_every", "_steps_since_retrain")

    _train_window: int
    _retrain_every: int
    _steps_since_retrain: int

    def __init__(
        self,
        train_window: int,
        retrain_every: int,
        inputs: tuple,
        shape: tuple[int, ...],
        dtype: np.dtype,
        state: Any,
    ) -> None:
        super().__init__(inputs, shape, dtype, state)
        self._train_window = train_window
        self._retrain_every = retrain_every
        self._steps_since_retrain = 0

    @abstractmethod
    def _fit(self, timestamp: np.datetime64, inputs: tuple) -> None:
        """Train the model on the input data."""

    @abstractmethod
    def _predict(self, timestamp: np.datetime64, inputs: tuple) -> ArrayLike | None:
        """Generate a prediction from the current model."""

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple, state: Any) -> ArrayLike | None:
        self._steps_since_retrain += 1
        if self._steps_since_retrain >= self._retrain_every:
            self._fit(timestamp, inputs)
            self._steps_since_retrain = 0
        return self._predict(timestamp, inputs)
