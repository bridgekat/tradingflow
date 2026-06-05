"""SimpleTrader leaf operator.

Ported from ``tradingflow.operators.traders.simple_trader``.  The
realistic execution path (dividend reinvestment, fees, lot rounding,
one-tick-delayed MOC execution, A-shares price limits, NAV valuation)
lives in ``flowops.traders._base.SimpleTrader``.

``SimpleTrader`` is parameterised by a ``trade_fn`` that maps the
current state + target soft positions to per-stock *lot* deltas.  The
upstream framework passed this as a constructor argument; here the Rust
host supplies config scalars via ``build(**kwargs)``.  Callers that need
a custom ``trade_fn`` pass it through ``build``; otherwise a default
value-weight trade function is used (treats the soft positions as target
market-value weights, sizing lots to hit those weights at the execution
price).

Inputs (array views, all shape ``(num_stocks,)``):
    0. soft positions
    1. close prices
    2. adjusts
    3. upper price limit
    4. lower price limit
Output: array view of shape ``(2,)`` -> ``(holdings_value, cash)``.

build kwargs:
    num_stocks (int, optional; else read from inputs[0].shape at init)
    initial_cash (float)
    lot_size (float)
    fee_base (float)
    fee_rate (float)
    verbose (bool, default False)
    trade_fn (callable, optional; default = value-weight targeting)
"""

from __future__ import annotations

import numpy as np

from flowops.traders._base import SimpleTrader, SimpleTraderState


def default_value_weight_trade_fn(
    state: SimpleTraderState,
    soft_positions: np.ndarray,
) -> np.ndarray:
    """Default ``trade_fn``: treat soft positions as target value weights.

    Sizes each tradable stock to ``weight * current_value`` at the
    execution price and returns the integer lot delta from the current
    holding.  Suspended / invalid-price stocks get zero.
    """
    N = state.num_stocks
    current_value = state._current_value
    exec_price = state._exec_price

    lots = np.zeros(N, dtype=np.float64)
    for i in range(N):
        p = exec_price[i]
        if not np.isfinite(p) or p <= 0:
            continue
        target_value = soft_positions[i] * current_value
        target_shares = target_value / p
        lots[i] = np.round((target_shares - state.shares[i]) / state.lot_size)
    return lots


def build(**kwargs) -> SimpleTrader:
    num_stocks = kwargs.get("num_stocks", None)
    trade_fn = kwargs.get("trade_fn", default_value_weight_trade_fn)
    return SimpleTrader(
        num_stocks=num_stocks,
        trade_fn=trade_fn,
        initial_cash=kwargs.get("initial_cash", 1_000_000.0),
        lot_size=kwargs.get("lot_size", 100.0),
        fee_base=kwargs.get("fee_base", 5.0),
        fee_rate=kwargs.get("fee_rate", 0.0003),
        verbose=kwargs.get("verbose", False),
    )
