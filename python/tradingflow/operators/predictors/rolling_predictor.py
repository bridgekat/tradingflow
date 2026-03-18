"""Abstract rolling predictor base class.

Provides [`RollingPredictor`][tradingflow.operators.predictors.RollingPredictor], an [`Operator`][tradingflow.Operator] subclass that
automatically retrains a model every *retrain_every* steps on a window
of the most recent *train_window* observations.  Subclasses implement
[`_fit`][tradingflow.operators.predictors.RollingPredictor._fit] (training) and [`_predict`][tradingflow.operators.predictors.RollingPredictor._predict] (inference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class RollingPredictor(Operator, ABC):
    """Abstract base for predictors that retrain on a rolling window.

    Subclasses must implement [`_fit`][._fit], [`_predict`][._predict], and
    [`init_state`][.init_state].  The predictor retrains every *retrain_every*
    steps, tracking the step count in the state dict under the key
    `"steps_since_retrain"`.

    Parameters
    ----------
    train_window
        Number of most-recent observations used for training.
    retrain_every
        Number of update steps between successive retraining calls.
    inputs
        Tuple of input series passed to [`_fit`][._fit] and [`_predict`][._predict].
    shape
        Element shape of the output series.
    dtype
        NumPy dtype for the output series.
    """

    __slots__ = ("_train_window", "_retrain_every")

    _train_window: int
    _retrain_every: int

    def __init__(
        self,
        train_window: int,
        retrain_every: int,
        inputs: tuple,
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> None:
        super().__init__(inputs, shape, dtype)
        self._train_window = train_window
        self._retrain_every = retrain_every

    @abstractmethod
    def _fit(self, timestamp: np.datetime64, inputs: tuple, state: Any) -> None:
        """Train the model on the input data, storing results in *state*."""

    @abstractmethod
    def _predict(self, timestamp: np.datetime64, inputs: tuple, state: Any) -> ArrayLike | None:
        """Generate a prediction from the current model in *state*."""

    @override
    def compute(self, timestamp: np.datetime64, inputs: tuple, state: Any) -> tuple[ArrayLike | None, Any]:
        state["steps_since_retrain"] += 1
        if state["steps_since_retrain"] >= self._retrain_every:
            self._fit(timestamp, inputs, state)
            state["steps_since_retrain"] = 0
        return self._predict(timestamp, inputs, state), state
