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

Prerequisites: Rust 1.95+ and Python 3.12+ (for the `python` feature).

```bash
uv sync
cargo build --features python
cargo run --example strategy_macd
```

`uv sync` creates `.venv/` and installs the operator implementations in `python/` into it as an editable install, together with NumPy and the rest of what those operators import.

Building the `python` feature involves finding and linking a Python interpreter, as described in [PyO3 Building and Distribution](https://pyo3.rs/main/building-and-distribution). It should be the one in `.venv/` — which is what PyO3 picks up from an activated virtual environment, and what the `PYO3_PYTHON` environment variable may be used to name explicitly. Whichever is chosen is remembered: the embedded interpreter starts up as a stand-in for it, and takes its standard library and its `site-packages` — and so the operator implementations — from wherever that interpreter lives. Neither `PYTHONHOME` nor `PYTHONPATH` needs to be set, and the built binary does not need the virtual environment activated to run.

On Windows, the `PATH` environment variable must contain the directory in which the Python interpreter's DLL is found, as mentioned in [PyO3 FAQ](https://pyo3.rs/main/faq.html#im-trying-to-call-python-from-rust-but-i-get-status_dll_not_found-or-status_entrypoint_not_found). For a virtual environment, that is the DLL of the installation it was created from rather than anything in the environment itself.

If NumPy operators are used, make sure to set `OPENBLAS_NUM_THREADS=1` when running: OpenBLAS is not thread-safe to use unless its internal parallelism is disabled (see [OpenBLAS FAQ](https://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications)).
