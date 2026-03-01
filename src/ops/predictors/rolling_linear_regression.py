"""Rolling linear regression predictor."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ... import Series
from .rolling_predictor import RollingPredictor


class RollingLinearRegression(RollingPredictor[np.float64]):
    """OLS linear regression retrained on a rolling window."""

    __slots__ = ("_fit_intercept",)

    def __init__(
        self,
        features: Series[Any],
        targets: Series[Any],
        train_window: int,
        retrain_every: int = 1,
        fit_intercept: bool = True,
    ) -> None:
        state: dict[str, Any] = {"coefficients": None, "intercept": None}
        super().__init__(
            train_window,
            retrain_every,
            [features, targets],
            state,
            np.dtype(np.float64),
            targets.shape,
        )
        self._fit_intercept = fit_intercept

    def _fit(self, timestamp: np.datetime64, *inputs: Series[Any]) -> None:
        features, targets = inputs[0], inputs[1]
        X = np.asarray(features.values[-self._train_window :], dtype=np.float64)
        y = np.asarray(targets.values[-self._train_window :], dtype=np.float64)
        n_samples = len(X)
        if n_samples < 2:
            return

        X_2d = X.reshape(n_samples, -1)
        y_2d = y.reshape(n_samples, -1)

        if self._fit_intercept:
            X_2d = np.hstack([X_2d, np.ones((n_samples, 1), dtype=np.float64)])

        beta, _, _, _ = np.linalg.lstsq(X_2d, y_2d, rcond=None)

        if self._fit_intercept:
            self._state["coefficients"] = beta[:-1]
            self._state["intercept"] = beta[-1]
        else:
            self._state["coefficients"] = beta
            self._state["intercept"] = None

    def _predict(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        if self._state["coefficients"] is None:
            return None
        features = inputs[0]
        x = np.asarray(features.values[-1], dtype=np.float64).reshape(1, -1)
        pred: NDArray[np.float64] = x @ self._state["coefficients"]
        if self._state["intercept"] is not None:
            pred = pred + self._state["intercept"]
        return pred.reshape(self.output.shape)
