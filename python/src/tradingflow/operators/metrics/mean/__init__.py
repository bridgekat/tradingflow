"""Mean-return prediction evaluators.

- [`InformationCoefficient`][tradingflow.operators.metrics.mean.information_coefficient.InformationCoefficient]
  —cross-sectional IC / RankIC between predicted scores and
  realized forward returns.
"""

from .information_coefficient import InformationCoefficient, InformationCoefficientState

__all__ = [
    "InformationCoefficient",
    "InformationCoefficientState",
]
