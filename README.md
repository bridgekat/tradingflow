# TradingFlow

[![Test](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml/badge.svg)](https://github.com/bridgekat/tradingflow/actions/workflows/test.yml)

A lightweight framework for quantitative investment research.

A trading strategy is a static computation graph: feature extraction, model prediction, portfolio optimization, trading simulation and performance evaluation are all operator nodes in this graph. Writing a strategy backtest amounts to wiring together reusable operators, and new operators can be readily implemented.

[Read the full documentation here.](https://bridgekat.github.io/tradingflow/)

This framework is structured into sub-packages in the `crates/` directory:

- The `tradingflow-data` package, which provides generic N-dimensional arrays and time series;
- The `tradingflow-graph` package, which provides abstractions and scheduling for generic computation graphs;
- The `tradingflow-macros` package, which provides procedural macros for composing subgraphs;
- The `tradingflow` package itself contains data loaders and operator implementations for quantitative investment research. Operators can additionally be written in Python and run on an embedded interpreter, giving strategies direct access to the data science ecosystem of Python.

## Get started

Prerequisites: Rust 1.95+ and Python 3.12+ (for the `python` feature).

```bash
uv sync
cargo build --features python
cargo run --example strategy_macd
```

The `python` feature links `libpython`, so it needs the environment variables `PYO3_PYTHON` and `PATH` to be set correctly. Moreover, the `PYTHONPATH` environment variable must point to `python/` in this repository, so that operator implementations can be imported.

If NumPy operators are used, make sure to set `OPENBLAS_NUM_THREADS=1` when running: OpenBLAS is not thread-safe to use unless its internal parallelism is disabled (see [OpenBLAS FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications)).
