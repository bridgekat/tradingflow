# TradingFlow

[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

A lightweight framework for quantitative investment research.

A trading strategy is a *static, stateful computation graph*: feature extraction, model prediction, portfolio optimization, trading simulation and performance evaluation are all operator nodes in this graph. When new data arrives, changes are automatically propagated throughout the graph. Writing a strategy backtest amounts to wiring together reusable operators, and new operators can be readily implemented.

[Read the full documentation here.](https://bridgekat.github.io/tradingflow/)

## Examples

Some examples can be found in [`crates/tradingflow/examples/`](crates/tradingflow/examples/). Run with:

```bash
git lfs pull  # Make sure market data files are available

cargo run --release --example demo
cargo run --release --example indicators
cargo run --release --example features
cargo run --release --example strategy_macd

cd crates/tradingflow/examples
uv sync  # Install dependencies from `pyproject.toml` using the `uv` package manager
. .venv/Scripts/activate  # Activate the virtual environment, so PyO3 links into it

PYTHONPATH=.venv/Lib/site-packages \
OPENBLAS_NUM_THREADS=1 \
cargo run --release --example strategy_factor --features python

python plot.py strategy_factor.csv  # Optional: plot the results
```

- [`demo`](crates/tradingflow/examples/demo.rs) is a simple MACD crossover strategy on synthetic price signals (the example in the docs).
- [`indicators`](crates/tradingflow/examples/indicators.rs) computes moving averages and Bollinger bands on real stock price history.
- [`features`](crates/tradingflow/examples/features.rs) evaluates the WorldQuant *101 Formulaic Alphas* catalog, logging cumulative daily IC (or RankIC with `--rank`) curves and printing per-feature mean IC and ICIR summaries.
- [`strategy_macd`](crates/tradingflow/examples/strategy_macd.rs) is the same strategy as `demo` tested on real data.
- [`strategy_factor`](crates/tradingflow/examples/strategy_factor.rs) demonstrates the use of Python operators to build alpha and risk models, with Markowitz portfolio optimization using CVXPY.

The examples come with a decade of daily price data for six A-shares stocks, stored as long-format CSV tables. They read the tables through the built-in CSV panel sources, forward-adjust closing prices for dividends, simulate frictionless trading, log results to a CSV output, and print summary statistics. Pass `-- --help` for available command-line options.

There is also a [`plot.py`](crates/tradingflow/examples/plot.py) which can be used to inspect the data and output CSV files. Pass `--help` for available command-line options.

## Python environment setup

The `tradingflow` crate provides the `python` feature flag, which enables Python operator support via [PyO3](https://github.com/PyO3/pyo3/). This needs to find and link against a Python interpreter, as described in [PyO3 Building and Distribution](https://pyo3.rs/main/building-and-distribution). On Windows, the `PATH` environment variable must contain the directory in which the Python interpreter's DLL is found, as mentioned in [PyO3 FAQ](https://pyo3.rs/main/faq.html#im-trying-to-call-python-from-rust-but-i-get-status_dll_not_found-or-status_entrypoint_not_found).

If a run requires Python operators using NumPy or SciPy, make sure to also set `OPENBLAS_NUM_THREADS=1`: OpenBLAS is not thread-safe to use unless its internal parallelism is disabled (see [OpenBLAS FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications)), otherwise the program may crash.

At run time, the Rust host must initialize the embedded interpreter via `pyo3::Python::initialize()`, before the first Python operator runs. The interpreter should be able to import NumPy, as well as any other dependencies used by the Python operators. This may be configured by e.g. setting the `PYTHONPATH` environment variable.
