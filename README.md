# TradingFlow

[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

A lightweight framework for quantitative investment research.

A trading strategy is a static computation graph: feature extraction, model prediction, portfolio optimization, trading simulation and performance evaluation are all operator nodes in this graph. Writing a strategy backtest amounts to wiring together reusable operators, and new operators can be readily implemented.

[Read the full documentation here.](https://bridgekat.github.io/tradingflow/)

This framework is structured into sub-packages in the `crates/` directory:

- The `tradingflow-data` package, which provides generic N-dimensional arrays and time series;
- The `tradingflow-graph` package, which provides abstractions and scheduling for generic computation graphs;
- The `tradingflow-macros` package, which provides procedural macros for composing operators;
- The `tradingflow` package itself contains data loaders and operator implementations for quantitative investment research. Operators can additionally be written in Python and run on an embedded interpreter, giving strategies direct access to the data science ecosystem of Python.

## Get started

Recommended setup: Rust 1.95+, Python 3.12+ and [the `uv` package manager](https://docs.astral.sh/uv/).

```bash
uv sync  # Create a virtual environment in `.venv/` and install dependencies
cargo build --features python  # Build `tradingflow` with the `python` feature enabled
```

Building with the `python` feature involves finding and linking a Python interpreter, as described in [PyO3 Building and Distribution](https://pyo3.rs/main/building-and-distribution). PyO3 prioritizes the one in `.venv/` if created; this can be overridden via the `PYO3_PYTHON` environment variable. On Windows, the `PATH` environment variable must contain the directory in which the Python interpreter's DLL is found, as mentioned in [PyO3 FAQ](https://pyo3.rs/main/faq.html#im-trying-to-call-python-from-rust-but-i-get-status_dll_not_found-or-status_entrypoint_not_found).

If a run requires Python operators using NumPy or SciPy, make sure to also set `OPENBLAS_NUM_THREADS=1`: OpenBLAS is not thread-safe to use unless its internal parallelism is disabled (see [OpenBLAS FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications)), which may crash the program.

## Additional examples

Simple strategy backtesting examples can be found in [`crates/tradingflow/examples/`](crates/tradingflow/examples/). Run with:

```bash
git lfs pull  # Make sure market data files are available
cargo run --release --example strategy_macd
cargo run --release --example strategy_macd_panel
cargo run --release --example strategy_markowitz_panel --features python
```

- `strategy_macd` is a simple MACD crossover strategy on synthetic price signals (the example in the docs).
- `strategy_macd_panel` is the same strategy as `strategy_macd` tested on real stock price history.
- `strategy_markowitz_panel` computes a few cross-sectional features (momentum, volatility, size), fits an incremental Ridge regression on them against next-day log returns alongside a shrinkage covariance estimate, and hands both moments to a simple Markowitz portfolio optimizer. The predictors and the optimizer are Python operators, so it needs the `python` feature and necessary dependencies installed.

The two `panel` examples backtest against real market history: a decade of daily price data for six A-shares stocks, stored as long-format CSV tables. Both read the tables through the built-in CSV panel sources, forward-adjust closing prices for dividends, simulate frictionless trading, log their NAV curves to a CSV under `target/`, and print summary statistics (return, volatility, Sharpe, max drawdown). Pass `-- --help` for available command-line options.
