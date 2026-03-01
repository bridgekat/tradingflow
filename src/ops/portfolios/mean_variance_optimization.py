"""Mean-variance optimization portfolio operator."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ... import Operator, Series


class MeanVarianceOptimization(Operator[None, np.float64]):
    """Mean-variance portfolio optimization placeholder."""

    __slots__ = ("_risk_aversion",)

    def __init__(
        self,
        predictions: Series[Any],
        covariances: Series[Any],
        risk_aversion: float = 1.0,
    ) -> None:
        raise NotImplementedError("MeanVarianceOptimization is not yet implemented")

    def compute(self, timestamp: np.datetime64, *inputs: Series[Any]) -> Optional[ArrayLike]:
        raise NotImplementedError
