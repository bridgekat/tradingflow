# TradingFlow

[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

A lightweight framework for quantitative investment research.

A trading strategy is a *static, stateful computation graph*: feature extraction, model prediction, portfolio optimization, trading simulation and performance evaluation are all operator nodes in this graph. When new data arrives, changes are automatically propagated throughout the graph. Writing a strategy backtest amounts to wiring together reusable operators, and new operators can be readily implemented.

[Read the full documentation here.](https://bridgekat.github.io/tradingflow/)

## Examples

Simple strategy backtesting examples can be found in [`crates/tradingflow/examples/`](crates/tradingflow/examples/). Run with:

```bash
git lfs pull  # Make sure market data files are available
cargo run --release --example strategy_macd
cargo run --release --example strategy_macd_panel
```

- `strategy_macd` is a simple MACD crossover strategy on synthetic price signals (the example in the docs).
- `strategy_macd_panel` is the same strategy as `strategy_macd` tested on real stock price history.

The `panel` example backtests against real market history: a decade of daily price data for six A-shares stocks, stored as long-format CSV tables. It reads the tables through the built-in CSV panel sources, forward-adjusts closing prices for dividends, simulates frictionless trading, logs its NAV curve to a CSV under `target/`, and prints summary statistics (return, volatility, Sharpe, max drawdown). Pass `-- --help` for available command-line options.

## Python environment setup

The `tradingflow` crate provides the `python` feature flag, which enables Python operator support via [PyO3](https://github.com/PyO3/pyo3/). This needs to find and link against a Python interpreter, as described in [PyO3 Building and Distribution](https://pyo3.rs/main/building-and-distribution). On Windows, the `PATH` environment variable must contain the directory in which the Python interpreter's DLL is found, as mentioned in [PyO3 FAQ](https://pyo3.rs/main/faq.html#im-trying-to-call-python-from-rust-but-i-get-status_dll_not_found-or-status_entrypoint_not_found).

If a run requires Python operators using NumPy or SciPy, make sure to also set `OPENBLAS_NUM_THREADS=1`: OpenBLAS is not thread-safe to use unless its internal parallelism is disabled (see [OpenBLAS FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications)), otherwise the program may crash.

At run time, the Rust host must initialize the embedded interpreter via `pyo3::Python::initialize()`, before the first Python operator runs. The interpreter should be able to import NumPy, as well as any other dependencies used by the Python operators. This may be configured by e.g. setting the `PYTHONPATH` environment variable.
