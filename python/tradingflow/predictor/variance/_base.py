"""Windowed covariance predictor harness.

Every estimator here ignores features and fits from the target panel alone, so
they all share one `predict`: the fitted matrix *is* the prediction. They
differ only in how the panel becomes a matrix.

Because none of them reads features, none of them retains them either: they
default `retain_features` to false and the harness skips the window entirely.
On a wide panel that is the difference between megabytes and gigabytes of
retained history per node. The flag is the same one the mean base carries — it
describes the fit, not the moment being predicted — so a factor-aware
covariance estimator could simply turn it back on.
"""

from dataclasses import dataclass

import numpy as np

from .._panel import PanelPredictor, PanelState


def predict(features: np.ndarray | None, params: np.ndarray) -> np.ndarray:
    """Returns the fitted covariance block unchanged.

    `features` is always `None` here: a covariance predictor is never handed a
    feature cross-section, because it never asked for one.
    """
    return params


@dataclass(slots=True)
class VariancePanelState(PanelState):
    """A covariance predictor emits a matrix over the active stocks."""

    @staticmethod
    def empty(n: int) -> np.ndarray:
        return np.full((n, n), np.nan)

    @staticmethod
    def scatter(out: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        out[np.ix_(mask, mask)] = values


class VariancePredictor(PanelPredictor):
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, np.ndarray]
    type Context = int
    type State = VariancePanelState

    state_type = VariancePanelState


def covariance_predictor(fit, *, retain_features: bool = False, **kwargs) -> VariancePredictor:
    """Wraps a `(T, N)` target panel estimator into a covariance predictor."""
    return VariancePredictor(
        fit=lambda x, y: fit(y),
        predict=predict,
        retain_features=retain_features,
        **kwargs,
    )
