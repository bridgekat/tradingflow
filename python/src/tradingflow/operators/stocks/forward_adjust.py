"""Forward price adjustment operator for corporate actions."""

from __future__ import annotations

import numpy as np

from ... import Handle, NativeOperator, NodeKind


class ForwardAdjust(NativeOperator):
    """Compute the forward-adjusted close price (or adjustment factor)
    for a single stock.

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
    output_prices
        If `True` (default), output the forward-adjusted price
        (`raw_price * factor`).  If `False`, output the cumulative
        adjustment factor itself.
    """

    def __init__(
        self,
        close: Handle,
        dividends: Handle,
        *,
        output_prices: bool = True,
    ) -> None:
        params: dict = {}
        if not output_prices:
            params["output_prices"] = False
        super().__init__(
            native_id="forward_adjust",
            inputs=(close, dividends),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(),
            params=params,
        )
