"""Cross-sectional linear regression predictor.

Provides [`CrossSectionalRegression`][tradingflow.operators.predictors.CrossSectionalRegression], a [`RollingPredictor`][tradingflow.operators.predictors.RollingPredictor]
that fits an OLS model across stocks (cross-sectionally) using realized
returns computed from historical prices.  No look-ahead bias: at time *t*
the model only uses returns that have already been realized.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike

from ... import Series
from .rolling_predictor import RollingPredictor


class CrossSectionalRegression(RollingPredictor):
    """Cross-sectional OLS regression retrained on a rolling window.

    At each retraining step the model collects historical factor values
    `(N, D)` paired with realized forward returns computed from the
    price series.  All (stock, timestamp) pairs within the training window
    whose forward returns are already known are stacked into a single
    regression problem.  NaN entries are filtered out before fitting.

    Parameters
    ----------
    features
        Vector-valued series of shape `(N, D)` — cross-sectional factors
        for *N* stocks and *D* factor dimensions.
    prices
        Vector-valued series of shape `(N,)` — close prices for *N* stocks.
    train_window
        Number of most-recent timestamps of historical data to use for
        training.  Each timestamp contributes up to *N* observations.
    retrain_every
        Number of update steps between successive retraining calls.
    return_horizon
        Number of timestamps defining the forward return period.
    """

    __slots__ = ("_return_horizon",)

    _return_horizon: int

    def __init__(
        self,
        features: Series,
        prices: Series,
        train_window: int = 10,
        retrain_every: int = 1,
        return_horizon: int = 21,
    ) -> None:
        n = features.shape[0]
        super().__init__(
            train_window,
            retrain_every,
            (features, prices),
            (n,),
            np.dtype(np.float64),
        )
        self._return_horizon = return_horizon

    @override
    def init_state(self) -> dict[str, Any]:
        return {"steps_since_retrain": 0, "coefficients": None}

    @override
    def _fit(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series, Series],
        state: dict[str, Any],
    ) -> None:
        features_series, prices_series = inputs
        n_features = len(features_series)
        n_prices = len(prices_series)
        if n_features < 2 or n_prices < self._return_horizon + 2:
            return

        # Determine usable range: timestamps where forward return is realized
        # We need prices at index i and i + return_horizon, both in the past
        usable_end = n_prices - self._return_horizon
        if usable_end <= 0:
            return

        n_use = min(n_features, usable_end, self._train_window)
        if n_use <= 0:
            return

        # Collect features and compute realized returns
        start_feat = n_features - n_use
        start_price = n_prices - self._return_horizon - n_use

        all_X: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for i in range(n_use):
            feat_idx = start_feat + i
            price_idx = start_price + i

            X_row = features_series[feat_idx]  # (N, D)
            p_now = prices_series[price_idx]  # (N,)
            p_future = prices_series[price_idx + self._return_horizon]  # (N,)

            returns = p_future / p_now - 1.0  # (N,)

            all_X.append(X_row)
            all_y.append(returns)

        X_stacked = np.vstack(all_X)  # (n_use * N, D)
        y_stacked = np.concatenate(all_y)  # (n_use * N,)

        # Filter NaN/inf rows
        valid = np.isfinite(y_stacked)
        for col in range(X_stacked.shape[1]):
            valid &= np.isfinite(X_stacked[:, col])

        X_clean = X_stacked[valid]
        y_clean = y_stacked[valid]

        if len(X_clean) < X_clean.shape[1] + 1:
            return

        X_bias = np.column_stack([X_clean, np.ones(len(X_clean))])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y_clean, rcond=None)
        state["coefficients"] = coeffs

    @override
    def _predict(
        self,
        timestamp: np.datetime64,
        inputs: tuple[Series, Series],
        state: dict[str, Any],
    ) -> ArrayLike | None:
        features_series, _ = inputs
        if state["coefficients"] is None or not features_series:
            return None
        X = features_series.values[-1]  # (N, D)
        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        predictions = X_bias @ state["coefficients"]
        # Set NaN for stocks with any NaN features
        nan_rows = np.any(~np.isfinite(X), axis=1)
        predictions[nan_rows] = np.nan
        return predictions
