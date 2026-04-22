"""Single-feature pass-through mean predictor."""

import numpy as np

from ..mean_predictor import MeanPredictor, MeanPredictorState


class SingleFeature(MeanPredictor[None]):
    """Return a single feature column directly as the prediction.

    A trivial pass-through predictor that selects one feature column
    and outputs it unchanged.  Useful for treating a raw factor as a
    "prediction" so it can be evaluated by
    [`InformationCoefficient`][tradingflow.operators.metrics.mean.information_coefficient.InformationCoefficient]
    at the predictor's rebalance cadence (rather than at the factor's
    native update cadence).

    No fitting is performed.  `max_periods` and `min_periods` are
    fixed to `1` since no historical window is needed — only the
    latest feature row is used.  The `target_series` input is still
    required (for alignment consistency with the base class) but is
    not consulted.

    Parameters
    ----------
    universe
        Universe weights, shape `(num_stocks,)`.
    features_series
        Recorded features series, element shape `(num_stocks, num_features)`.
    target_series
        Recorded target series, element shape `(num_stocks,)`.
        Passed through but not used.
    feature_index
        Index of the feature column to return.  Default `0`.
    **kwargs
        Forwarded to [`MeanPredictor`][tradingflow.operators.predictors.mean_predictor.MeanPredictor].
        `max_periods` and `min_periods` are fixed to `1`.
    """

    def __init__(
        self,
        universe,
        features_series,
        target_series,
        *,
        feature_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            universe,
            features_series,
            target_series,
            fit_fn=_fit_fn,
            predict_fn=lambda state, features, params: features[:, feature_index],
            max_periods=1,
            min_periods=1,
            **kwargs,
        )


def _fit_fn(x: np.ndarray, y: np.ndarray) -> None:
    """No-op: SingleFeature does not fit any parameters."""
    return None
