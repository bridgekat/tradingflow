[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

**TradingFlow** is a lightweight library for quantitative investment research that supports multi-frequency market data, formulaic factors, forecasting models, portfolio optimization methods and backtesting in a unified data model. The core runtime is implemented in Rust; a Python wrapper is provided for ease of use with Python's data science ecosystem.

Main design goals:

- **Composable modules:** trading strategies are computation graphs, whose nodes are either data sources or operators. Common sources and operators are provided out of the box, and new ones can be readily implemented in either Rust or Python.
- **Agent-friendly codebase:** we maintain code-documentation consistency and a hierarchy of documented modules to facilitate LLM code exploration and generation. When using LLM coding agents (Claude Code, Codex, OpenCode, etc.), start every session by instructing the agent to read [AGENTS.md](AGENTS.md) and then describe your tasks.

# Get started

Prerequisites: Python 3.12+ and a stable Rust toolchain ([rustup.rs](https://rustup.rs)).

TradingFlow uses [Maturin](https://www.maturin.rs/) to compile the Rust core and install the Python package together. Clone the repository and install in editable mode:

```bash
git clone https://github.com/bridgekat/tradingflow.git
cd tradingflow
pip install -e ".[dev]"
```

A minimal scenario builds a graph consisting of source and operator nodes. Each `add_source` / `add_operator` returns a handle that downstream operators consume; `run()` drives the event loop:

```python
import numpy as np
import tradingflow as tf

sc = tf.Scenario()

timestamps = np.arange("2024-01-01", "2024-04-01", dtype="datetime64[D]")
values = np.random.randn(len(timestamps)).cumsum() + 100.0
prices = sc.add_source(tf.sources.ArraySource(timestamps, values, dtype=np.float64))

history = sc.add_operator(tf.operators.Record(prices))
sc.run()

print(sc.series_view(history).to_series().tail())
```

# Examples

To run the shipped example scripts, install the `examples` extras first:

```bash
pip install -e ".[examples]"
```

The [`python/examples/`](python/examples/) directory contains examples that load A-shares market data (requires manual download via the companion [a-shares-crawler](https://github.com/bridgekat/a-shares-crawler)) and run full strategy pipelines. After installing the `examples` extras, run `python -m a_shares_crawler --help` for configuration and download instructions.

- [**Plotting daily prices**](python/examples/plot_daily_price.py) — loads daily price history, computes forward-adjusted prices, moving average and Bollinger Bands, and plots them.
- [**Plotting financial data**](python/examples/plot_financial_data.py) — loads equity structure, balance sheet, income statement and cash flow data, computes market cap and annualized financial metrics, and plots them.
- [**Plotting total market cap**](python/examples/plot_total_market_cap.py) — loads daily prices and equity structures for all stocks, computes per-stock circulating market cap, and plots the total across the entire market over time.
- [**Mean strategy backtesting**](python/examples/mean_strategy.py) — loads daily prices, equity structures, dividends, and financial reports for all stocks, computes cross-sectional features, periodically fits a linear regression to predict stock returns, selects a portfolio from the top-predicted stocks with rank-linear weights, simulates trading with transaction costs, and plots portfolio value, rolling Sharpe ratio and drawdown against a market-cap-weighted index.
- [**Mean-variance strategy backtesting**](python/examples/mean_variance_strategy.py) — extends the mean strategy with Ledoit-Wolf shrinkage covariance estimation and Markowitz mean-variance portfolio optimisation (via CVXPY), comparing multiple risk-aversion levels against the index.
- [**Mean factor evaluation**](python/examples/factor_ic.py) — computes daily cross-sectional factors (log market cap, log book-to-price, turnover MA), evaluates each factor's predictive power via Information Coefficient (Pearson or Spearman), and plots cumulative IC curves.
- [**Covariance estimator comparison**](python/examples/covariance_gmv.py) — compares sample covariance and Ledoit-Wolf shrinkage estimators by measuring the realized variance of their respective Global Minimum Variance portfolios over time.

# Documentation

Read the full documentation [here](https://bridgekat.github.io/tradingflow/).

Currently, most documentations are LLM-generated, and may not be concise enough for human readers. The situation will be improved after core modules are stabilized.
