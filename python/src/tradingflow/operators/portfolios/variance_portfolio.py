"""Abstract variance portfolio operator."""

from __future__ import annotations

from typing import Callable
from dataclasses import dataclass

import numpy as np

from ... import ArrayView, Handle, NodeKind, Operator


@dataclass(slots=True)
class VariancePortfolioState:
    """Mutable state for [`VariancePortfolio`][tradingflow.operators.portfolios.variance_portfolio.VariancePortfolio] subclasses."""

    num_stocks: int
    logarithmic: bool
    positions_fn: Callable[["VariancePortfolioState", np.ndarray], np.ndarray]


class VariancePortfolio(
    Operator[
        ArrayView[np.float64],
        ArrayView[np.float64],
        ArrayView[np.float64],
        VariancePortfolioState,
    ]
):
    """Abstract portfolio constructor from covariance alone (no expected returns).

    Triggered by `universe` updates — the universe is the canonical
    rebalance signal.  On each trigger, delegates to `positions_fn` to
    compute position weights.  Only stocks with positive universe
    weights and finite diagonal covariance entries are passed to
    `positions_fn`; the result is scattered back to the full dimension
    with zeros elsewhere.

    The rebalance cadence is inherited from upstream: `universe` is
    typically clocked by a rebalance clock (e.g. via
    [`Clocked`][tradingflow.operators.clocked.Clocked]), so this operator runs
    at that cadence.  `predicted_covariances` is read as the latest
    stored prediction at the trigger — it need not produce on the same
    cycle.

    ## Expected prediction semantics

    Controlled by `logarithmic`:

    - `logarithmic=True` (default): `predicted_covariances` is a
      **log-return** covariance matrix.  The base class derives an
      implied linear-return mean and covariance via the zero-mean
      specialisation of the lognormal moment map before calling
      `positions_fn`:

          mu_lin[i]       = exp(½ Sigma_log[i, i]) - 1
          Sigma_lin[i, j] = (1 + mu_lin[i]) (1 + mu_lin[j])
                            · (exp(Sigma_log[i, j]) - 1)

    - `logarithmic=False`: `predicted_covariances` is already a
      **linear-return** covariance matrix and is forwarded to
      `positions_fn` unchanged.

    Either way, `positions_fn` always sees a linear-return covariance.
    At daily or weekly frequency, log-return covariance entries are
    small and the conversion is close to the identity; using the
    correct formula keeps the numbers consistent with
    [`MeanVariancePortfolio`][tradingflow.operators.portfolios.mean_variance_portfolio.MeanVariancePortfolio],
    which applies the full two-moment conversion.

    ## NaN behavior

    `predicted_covariances` is allowed to contain `NaN` rows and columns
    — per the
    [`VariancePredictor`][tradingflow.operators.predictors.variance_predictor.VariancePredictor]
    contract, these mark stocks with insufficient data.  The base class
    subsets to `(universe > 0) & np.isfinite(np.diag(Sigma))` before
    calling `positions_fn`, so subclasses never see `NaN` inputs.  The
    sub-covariance-matrix of the remaining stocks must not contain
    non-finite off-diagonal entries (the base class raises
    `ValueError` if it does).  The emitted position vector is zero for
    stocks outside the valid subset.

    Parameters
    ----------
    universe
        Handle to universe weights, shape `(num_stocks,)`.
        Stocks with positive values are included in the optimization.
    predicted_covariances
        Handle to predicted covariance matrix, shape
        `(num_stocks, num_stocks)`.  Log or linear depending on
        `logarithmic`.
    positions_fn
        `(state, Sigma) -> weights`.  Receives only the universe-active
        sub-block of the covariance matrix, in linear-return units.
    logarithmic
        If `True` (default), `predicted_covariances` is interpreted as
        a log-return covariance and converted via the zero-mean
        lognormal moment map before the inner call; if `False`, it is
        interpreted as a linear-return covariance and passed through
        unchanged.
    """

    def __init__(
        self,
        universe: Handle,
        predicted_covariances: Handle,
        *,
        positions_fn: Callable[[VariancePortfolioState, np.ndarray], np.ndarray],
        logarithmic: bool = True,
    ) -> None:
        assert len(universe.shape) == 1
        assert len(predicted_covariances.shape) == 2
        assert predicted_covariances.shape[0] == universe.shape[0]
        assert predicted_covariances.shape[1] == universe.shape[0]

        self._num_stocks = universe.shape[0]
        self._logarithmic = logarithmic
        self._positions_fn = positions_fn

        inputs = (universe, predicted_covariances)
        super().__init__(
            inputs=inputs,
            kind=NodeKind.ARRAY,
            dtype=np.float64,
            shape=(self._num_stocks,),
            name=type(self).__name__,
        )

    def init(
        self,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        timestamp: int,
    ) -> VariancePortfolioState:
        return VariancePortfolioState(
            num_stocks=self._num_stocks,
            logarithmic=self._logarithmic,
            positions_fn=self._positions_fn,
        )

    @staticmethod
    def compute(
        state: VariancePortfolioState,
        inputs: tuple[ArrayView[np.float64], ArrayView[np.float64]],
        output: ArrayView[np.float64],
        timestamp: int,
        produced: tuple[bool, ...],
    ) -> bool:
        # Trigger on universe updates: the universe is the canonical
        # rebalance signal, and the predictor's sigma is stored as the
        # last prediction even when it did not produce this cycle.
        if not produced[0]:
            return False

        universe = inputs[0].value()
        sigma = inputs[1].value()

        mask = (universe > 0) & np.isfinite(np.diag(sigma))
        sub_sigma = sigma[np.ix_(mask, mask)]
        if not np.all(np.isfinite(sub_sigma)):
            raise ValueError("sub-covariance matrix contains non-finite entries")
        # Lognormal conversion (zero-mean specialisation):
        #   mu_lin[i]       = exp(0.5 * Sigma_log[i, i]) - 1
        #   Sigma_lin[i, j] = (1 + mu_lin[i])(1 + mu_lin[j])
        #                     * (exp(Sigma_log[i, j]) - 1)
        if state.logarithmic:
            sub_mu = np.expm1(0.5 * np.diag(sub_sigma))
            factor = 1.0 + sub_mu
            sub_sigma = np.outer(factor, factor) * np.expm1(sub_sigma)

        positions = np.zeros_like(universe, dtype=np.float64)
        if mask.any():
            positions[mask] = state.positions_fn(state, sub_sigma)

        output.write(positions)
        return True
