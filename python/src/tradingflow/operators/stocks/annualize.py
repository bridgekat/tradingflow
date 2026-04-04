"""Annualize operator for YTD financial report data."""

from __future__ import annotations

import numpy as np

from ...operator import NativeOperator
from ...types import Handle, NodeKind


class Annualize(NativeOperator):
    """Convert year-to-date financial values into annualized values.

    Takes an input array of shape `(2 + N,)` laid out as
    `[year, day_of_year, ytd_1, ..., ytd_N]` (produced by
    [`FinancialReportSource`][tradingflow.sources.FinancialReportSource] with
    `with_report_date=True`) and outputs an array of shape `(N,)`
    containing annualized values.

    The operator handles any reporting frequency uniformly by using
    days-based scaling: `annualized = (current - last) × 365 / days_elapsed`.

    Parameters
    ----------
    source
        Handle to a `FinancialReportSource` output with
        `with_report_date=True`.
    """

    def __init__(self, source: Handle) -> None:
        # Input shape is (2 + N,); output shape is (N,).
        input_stride = 1
        for d in source.shape:
            input_stride *= d
        n = input_stride - 2
        if n < 1:
            raise ValueError(
                f"Annualize: input shape {source.shape} must have at least 3 elements "
                f"([year, day_of_year, ...values])"
            )
        output_shape = (n,) if n > 1 else ()

        super().__init__(
            native_id="annualize",
            inputs=(source,),
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=output_shape,
        )
