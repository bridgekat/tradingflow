"""RandomTrader leaf operator.

Ported from ``tradingflow.operators.traders.simple.random_trader``.

On each rebalance, picks ``portfolio_size`` stocks (without replacement)
using the soft position weights as sampling probabilities, then
allocates equal market-value weight ``1 / portfolio_size`` to each
chosen stock.  Built on top of ``flowops.traders._base.SimpleTrader``
(same realistic execution path); only the ``trade_fn`` differs.

Inputs (array views, all shape ``(num_stocks,)``):
    0. soft positions
    1. close prices
    2. adjusts
    3. upper price limit
    4. lower price limit
Output: array view of shape ``(2,)`` -> ``(holdings_value, cash)``.

build kwargs:
    num_stocks (int, optional; else read from inputs[0].shape at init)
    portfolio_size (int, default 20)
    initial_cash (float)
    lot_size (float)
    fee_base (float)
    fee_rate (float)
    verbose (bool, default False)
    seed (int, optional; seeds the RNG for reproducibility)
"""

from __future__ import annotations

import numpy as np

from flowops.traders._base import SimpleTrader, SimpleTraderState


def _trade_fn(
    state: SimpleTraderState,
    soft_positions: np.ndarray,
    portfolio_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    N = state.num_stocks
    current_value = state._current_value
    exec_price = state._exec_price

    # Exclude stocks with no valid execution price (e.g. suspended).
    tradable = np.isfinite(exec_price) & (exec_price > 0)

    # Normalize soft weights into sampling probabilities.
    weights = np.where(tradable, np.maximum(soft_positions, 0.0), 0.0)
    s = weights.sum()
    if s <= 0:
        return np.zeros(N, dtype=np.float64)
    weights /= s

    # Randomly select portfolio_size stocks (or fewer if not enough candidates).
    n_candidates = (weights > 0).sum()
    n_select = min(portfolio_size, n_candidates)
    if n_select == 0:
        return np.zeros(N, dtype=np.float64)

    chosen = rng.choice(N, n_select, replace=False, p=weights)

    # Equal-weight hard positions for chosen stocks.
    hard_positions = np.zeros(N, dtype=np.float64)
    hard_positions[chosen] = 1.0 / portfolio_size * s

    # Convert to lot counts.
    lots = np.zeros(N, dtype=np.float64)
    for i in range(N):
        p = exec_price[i]
        if not np.isfinite(p) or p <= 0:
            continue
        target_value = hard_positions[i] * current_value
        target_shares = target_value / p
        lots[i] = np.round((target_shares - state.shares[i]) / state.lot_size)

    return lots


def build(**kwargs) -> SimpleTrader:
    portfolio_size = kwargs.get("portfolio_size", 20)
    seed = kwargs.get("seed", None)
    rng = np.random.default_rng(seed)
    trade_fn = lambda state, sp: _trade_fn(state, sp, portfolio_size, rng)
    return SimpleTrader(
        num_stocks=kwargs.get("num_stocks", None),
        trade_fn=trade_fn,
        initial_cash=kwargs.get("initial_cash", 1_000_000.0),
        lot_size=kwargs.get("lot_size", 100.0),
        fee_base=kwargs.get("fee_base", 5.0),
        fee_rate=kwargs.get("fee_rate", 0.0003),
        verbose=kwargs.get("verbose", False),
    )
