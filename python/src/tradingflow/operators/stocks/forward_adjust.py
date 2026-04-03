"""Forward price adjustment operator for corporate actions."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle


class ForwardAdjust(NativeOperator):
    """Compute the forward-adjusted close price for a single stock.

    Takes a scalar close price and a dividend vector `[share_dividends,
    cash_dividends]`.  When the dividend input updates, the operator
    computes an adjustment multiplier using the previous close price and
    accumulates it into a cumulative factor.

    Parameters
    ----------
    close
        Scalar `Array<float64>` handle for the close price.
    dividends
        `Array<float64>` handle of shape `(2,)`:
        `[share_dividends, cash_dividends]`.
    """

    def __init__(self, close: Handle, dividends: Handle) -> None:
        super().__init__(
            kind="forward_adjust",
            inputs=(close, dividends),
            shape=(),
            dtype=np.float64,
        )
