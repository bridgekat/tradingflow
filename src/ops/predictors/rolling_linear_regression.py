"""Rolling linear regression predictor.

Provides :class:`RollingLinearRegression`, a :class:`RollingPredictor`
that fits an OLS model with bias on vector features and scalar targets.
The model is retrained on the last *train_window* aligned observations
every *retrain_every* update steps.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling_predictor import RollingPredictor


class RollingLinearRegression(RollingPredictor):
    """OLS linear regression retrained on a rolling window.

    Features are a vector-valued series and targets a scalar series.
    At each step the model predicts ``X @ coefficients`` for the latest
    feature vector.
    """

    __slots__ = ()

    def __init__(
        self,
        features: Series,
        targets: Series,
        train_window: int = 10,
        retrain_every: int = 1,
    ) -> None:
        state: dict[str, Any] = {"coefficients": None}
        super().__init__(
            train_window,
            retrain_every,
            (features, targets),
            (),
            np.dtype(np.float64),
            state,
        )

    @override
    def _fit(self, timestamp: np.datetime64, inputs: tuple[Series, Series]) -> None:
        features, targets = inputs
        if len(features) < 2 or len(targets) < 2:
            return
        n = min(len(features), len(targets), self._train_window)
        X = features.values[-n:]
        y = targets.values[-n:]
        X_bias = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        self._state["coefficients"] = coeffs

    @override
    def _predict(self, timestamp: np.datetime64, inputs: tuple[Series, Series]) -> ArrayLike | None:
        features, targets = inputs
        if self._state["coefficients"] is None or not features:
            return None
        x = features.values[-1]
        X_bias = np.append(x, 1.0)
        return float(X_bias @ self._state["coefficients"])
