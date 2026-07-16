# TradingFlow

[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

A lightweight framework for quantitative investment research.

A trading strategy is a static computation graph: feature extraction, model prediction, portfolio optimization, trading simulation and performance evaluation are all operator nodes in this graph. Writing a strategy backtest amounts to wiring together reusable operators, and new operators can be readily implemented.

This framework is structured into sub-packages in the `crates/` directory:

- The [`tradingflow-data`](crates/tradingflow-data/README.md) package, which provides generic N-dimensional arrays and time series;
- The [`tradingflow-graph`](crates/tradingflow-graph/README.md) package, which provides abstractions and scheduling for generic computation graphs;
- The [`tradingflow-macros`](crates/tradingflow-macros/README.md) package, which provides procedural macros for composing sub-graphs;
- The [`tradingflow`](crates/tradingflow/README.md) package itself contains operator implementations for quantitative investment research. Operators can additionally be written in Python and run on an embedded interpreter, giving strategies direct access to the data science ecosystem of Python.

## Get started

Prerequisites: a stable Rust toolchain ([rustup.rs](https://rustup.rs)), and Python 3.12+ for the `python` feature.

```bash
git clone https://github.com/bridgekat/tradingflow.git
cd tradingflow
cargo build -p tradingflow --features python
```

The `python` feature links `libpython`, so it needs the environment variables `PYO3_PYTHON` and `PATH` to be set correctly. Moreover, the `PYTHONPATH` environment variable must point to `python/` in this repository, so that operator implementations can be imported.

## Examples

The [`examples/`](examples/) directory contains end-to-end strategies that load A-shares market data and run full pipelines (see [`examples/README.md`](examples/README.md) for build/run instructions). To follow along, install the `examples` extras (which fetch the [a-shares-crawler](https://github.com/bridgekat/a-shares-crawler) from GitHub) and download data:

```bash
pip install -e ".[examples]"       # fetches a-shares-crawler from GitHub + matplotlib
python -m a_shares_crawler --help  # For configuration & download instructions
```

The crawler writes one CSV per symbol by default; pass `--export-long {csv,parquet}...` to emit consolidated long-format tables which the examples read.

**Visualizations**:

- [**Daily prices**](examples/plot_daily_price.rs) — loads daily prices, computes forward-adjusted prices, a moving average, and Bollinger Bands.
- [**Financial data**](examples/plot_financial_data.rs) — loads equity structure, balance sheet, income statement, and cash flow data; computes market cap and annualized metrics.
- [**Total market cap**](examples/plot_total_market_cap.rs) — builds a cap-weighted A-shares index (top-N by circulating market cap, periodically rebalanced) and plots the sum of constituents' circulating market cap alongside the index's total-return NAV curve.

**Research utilities**:

- [**Factor IC**](examples/factor_ic.rs) — computes daily cross-sectional factors (log market cap, log book-to-price, turnover MA), evaluates each one's predictive power using ICs (information coefficients: the rank correlation coefficients), plots cumulative IC curves, and reports their respective IRs (information ratios: the Sharpe ratios of IC curves).
- [**Variance estimator comparison**](examples/covariance_gmv.rs) — compares the sample covariance estimator with and without Ledoit-Wolf shrinkage, by measuring the realized variance of their respective GMV (global minimum variance) portfolios.

**Backtests**:

- [**Mean-only strategy**](examples/mean_strategy.rs) — fits a periodic linear regression to predict cross-sectional stock returns, picks the top-ranked names with rank-linear weights, simulates trading with transaction costs, and plots portfolio value, rolling Sharpe, and drawdown vs. a market-cap-weighted index.
- [**Mean-variance strategy**](examples/mean_variance_strategy.rs) — extends the mean strategy with Ledoit-Wolf shrinkage covariance estimator and Markowitz portfolio optimization, comparing several risk-aversion levels.
