# Running the examples

The `.rs` files in this directory are end-to-end A-shares research strategies
built on the `flowgraph` engine (through the `tradingflow` operator library and
the `scenario` driver). They are ordinary Cargo examples:

```
cargo run --example <name> [--features python] -- [args]
```

There are two kinds:

* **Native examples** (`plot_daily_price`, `plot_financial_data`,
  `plot_total_market_cap`) are pure Rust — no Python toolchain, no extra setup.
  `plot_total_market_cap` is a full cap-weighted index backtest (executor
  included) on the native operator set.
* **`python` examples** embed a CPython interpreter and call the optional
  `flowops` operators (predictors / portfolios / a few metrics) from Rust, with
  real NumPy / SciPy / cvxpy. (The traders and core performance metrics are
  native Rust.) These need a configured Python environment — most of this guide
  is about that. The "shared library not found" and "`flowops` not found" errors
  are environment misconfiguration, not bugs in the examples.

Every example prints its arguments with `--help`. The cross-sectional examples
**require** `--data-dir`, `--begin`, `--end`, `--index-size`, and
`--rebalance-days` (the feature/strategy ones also require `--window`); the
single-stock plots take a `SYMBOL`. There are no hidden defaults — run with
`--help` to see them.

---

## Quickstart

A **native** example needs no Python — just choose a date range and universe
size (paths use forward slashes, which work on every platform):

```
cargo run --example plot_total_market_cap -- \
  --data-dir examples/data --begin 2023-01-01 --end 2024-12-31 \
  --index-size 1000 --rebalance-days 90
```

A **`python`** example needs a virtual environment and three environment
variables pointing the embedded interpreter at it:

```sh
# Linux / macOS (bash)
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[examples]"

export PYO3_PYTHON="$PWD/.venv/bin/python"            # build: which interpreter PyO3 links
cargo run --example mean_variance_strategy --features python -- \
  --data-dir examples/data --begin 2023-01-01 --end 2024-12-31 \
  --index-size 1000 --rebalance-days 90 --window 20
```

```powershell
# Windows (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[examples]"

. .\examples\env.ps1   # sets the three variables (PYO3_PYTHON, DLL dir, PYTHONPATH) for this shell
$common = '--data-dir','examples/data','--begin','2023-01-01','--end','2024-12-31','--index-size','1000','--rebalance-days','90'
cargo run --example mean_variance_strategy --features python -- @common --window 20
```

`examples/env.ps1` is a Windows convenience that exports the variables described
below for the current shell (dot-source it: note the leading `.` + space). There
is no Unix equivalent script — export the variables yourself (or from a shell
profile). The next section explains what they do.

---

## How the embedding works (why the environment matters)

The `flowgraph` engine has **no Python-as-host API**: graphs are built and driven
from Rust, and Python operators run *inside* an embedded CPython via PyO3
(`pyo3/auto-initialize`, Cargo feature `python`). Two distinct moments need
configuration:

1. **Build time** — PyO3 links the binary against one specific interpreter,
   chosen by **`PYO3_PYTHON`** (an absolute path to a `python` executable). That
   fixes which `libpython` the binary loads and which ABI it expects. If unset,
   PyO3 picks whatever `python3` is on `PATH`, which may be the wrong version or
   a stub with no development libraries.
2. **Run time** — the embedded interpreter must be able to (a) *load* its shared
   library and (b) *import* the operators' dependencies. PyO3 does **not**
   activate the venv, so:
   * the directory containing `libpython` must be on the OS library search path;
   * `flowops` and the venv's `site-packages` must be importable.

**Cardinal rule:** `PYO3_PYTHON`, the `libpython` you load, and the
`site-packages` you import must all belong to the **same** interpreter. Mixing
(say) a GIL build with free-threaded NumPy yields cryptic import / load failures.

### The three knobs

| Variable | When | What it points to |
|---|---|---|
| `PYO3_PYTHON` | build | the venv's interpreter (`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows) |
| OS library path | run | the directory holding `libpython` (see the per-OS table below) |
| `PYTHONPATH` | run | the repo's `python/` dir (for `flowops`) plus the venv's `site-packages` — *or* nothing, if you installed the project editable so `flowops` is already in `site-packages` |

The OS library path is platform-specific:

| OS | Variable | `libpython` filename |
|---|---|---|
| Windows | prepend to `PATH` | `pythonXY.dll` (e.g. `python313.dll`) |
| Linux | `LD_LIBRARY_PATH` | `libpythonX.Y.so` |
| macOS | `DYLD_LIBRARY_PATH` | `libpythonX.Y.dylib` |

`libpython` often lives in the venv's **base prefix** (the `home = …` line of
`.venv/pyvenv.cfg`), not the venv itself — especially for managed / standalone
Python installs. `env.ps1` reads that line so you don't hardcode it; on any
platform you can locate the directory with:

```
<python> -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
```

On Linux and macOS the venv interpreter can often resolve `libpython` on its own
(via an embedded rpath / `ldconfig`), so you only need the library-path variable
if you actually hit a load error.

### Setting the knobs by hand

```sh
# Linux / macOS
repo="$PWD"; py="$repo/.venv/bin/python"
export PYO3_PYTHON="$py"
export PYTHONPATH="$repo/python:$("$py" -c 'import sysconfig;print(sysconfig.get_path("purelib"))')"
# only if the embedded interpreter can't load libpython on its own:
export LD_LIBRARY_PATH="$("$py" -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH"
#   macOS: use DYLD_LIBRARY_PATH instead of LD_LIBRARY_PATH
```

```powershell
# Windows — equivalent to `. .\examples\env.ps1`
$repo = (Get-Location).Path
$env:PYO3_PYTHON = "$repo\.venv\Scripts\python.exe"
$base = (Select-String -Path "$repo\.venv\pyvenv.cfg" -Pattern '^home\s*=\s*(.+)$').Matches[0].Groups[1].Value.Trim()
$env:PATH = "$base;$env:PATH"
$env:PYTHONPATH = "$repo\python;$repo\.venv\Lib\site-packages"
```

---

## Setting up the virtual environment

Any CPython **3.12+** works (the project requires `>=3.12`). Use a real
interpreter, not an OS app-store stub.

```sh
python3 -m venv .venv                          # Windows: py -3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip  # Windows: .\.venv\Scripts\python -m pip ...
.venv/bin/python -m pip install -e ".[examples]"
```

`".[examples]"` installs the project's runtime dependencies (NumPy, SciPy, cvxpy
and its solvers, pandas), the `flowops` package (editable, so it's importable),
the `a-shares-crawler` (fetched from GitHub), and matplotlib for the plot
scripts. For just the operator dependencies without the crawler / plotting, use
`pip install -e .`.

Verify the environment (prints `env OK`):

```sh
.venv/bin/python -c "import numpy, scipy, cvxpy, matplotlib; import flowops.portfolios.mean_variance.markowitz; print('env OK')"
```

> The bare `python` / `python3` on your `PATH` may not be the venv's interpreter
> (on Windows it is often an app-store stub). Invoke the venv explicitly for the
> plot scripts, and let `PYO3_PYTHON` select the interpreter for builds.

---

## Getting the data

The examples read A-shares market data from `examples/data/`: consolidated
long-format `<kind>.parquet` tables plus `symbol_list.csv` (`a_shares_history/`,
if present, holds the legacy one-CSV-per-symbol layout, no longer read). The
crawler that produces it
([a-shares-crawler](https://github.com/bridgekat/a-shares-crawler)) is fetched
from GitHub by the `examples` extra installed above.

Fetching data needs an EastMoney session config — see the
[a-shares-crawler README](https://github.com/bridgekat/a-shares-crawler) for the
full procedure. In short:

```
.venv/bin/python -m a_shares_crawler --config config.json --data-dir examples/data
```

The crawler can emit **consolidated long-format tables** (one file per kind, all
symbols, sorted by date) for a single sequential read, via
`--export-long {csv,parquet}…` (or the standalone `python -m a_shares_crawler.export`):

```
.venv/bin/python -m a_shares_crawler.export --data-dir examples/data --export-long parquet
```

The examples read these long tables through the cross-sectional
`ParquetPanelSource` / `ReportPanelSource`.

---

## The examples

In the commands below, **`<common>`** stands for the shared required arguments:

```
--data-dir examples/data --begin 2023-01-01 --end 2024-12-31 --index-size 1000 --rebalance-days 90
```

`--index-size N` bounds the cap-weighted universe (smaller is faster); trim
`--begin` / `--end` for quick smoke tests. In PowerShell you can hold these in an
array and splat them as `@common` (as in the Quickstart). Run the plot scripts
with the **venv's** Python so matplotlib is found.

All examples read the consolidated long parquet tables
(`<data-dir>/<kind>.parquet`) through `ParquetPanelSource` / `ReportPanelSource`.

### Native (no `python` feature, no Python)

| Example | Command | Plot |
|---|---|---|
| `plot_daily_price` | `cargo run --example plot_daily_price -- 000009.SZ` | `python examples/plot.py target/plot_daily_price.csv` |
| `plot_financial_data` | `cargo run --example plot_financial_data -- 000001.SZ` | `python examples/plot_financial_data.py target/plot_financial_data.csv` |
| `plot_total_market_cap` | `cargo run --example plot_total_market_cap -- <common>` | `python examples/plot_total_market_cap.py target/plot_total_market_cap.csv` |

`ParquetPanelSource` pivots one long table into a wide `[N_symbols, K]`
cross-section per date; the per-stock pipeline is recovered by `Select`ing one
stock's row + a NaN `Filter`. The single-stock plots therefore read the whole
table (including the ~17.5M-row `daily_prices`) to extract one stock, so a run
takes ~1–2 min — the `1 → N → 1` fan-out is intentional for now.

**Semantics:** each cross-section reflects only that date's rows (absent symbols
`NaN`); the panel does **not** carry values forward or seed the window — the
"carry last value" / "NaN-fill" is the downstream `Stack` / `StackSync`'s job, as
with the old per-symbol sources. Because irregular kinds (dividends, reports)
tick on the *union* of all stocks' event dates (not every trading day), the
per-stock `Filter` drops the all-`NaN` ticks so message-passing operators like
`ForwardAdjust` see each real event once. `ReportPanelSource` aligns reports on
the **effective date** `max(report, notice)` (`use_effective_date`) so backtests
don't see a report before it was published.

### `python` feature, NumPy only (any GIL venv with NumPy works)

| Example | Command | Plot |
|---|---|---|
| `mean_strategy` | `cargo run --example mean_strategy --features python -- <common> --window 20` | `python examples/plot_strategy.py target/mean_strategy.csv` |
| `factor_ic` | `cargo run --example factor_ic --features python -- <common> --window 20` | `python examples/plot_factor_ic.py target/factor_ic.csv` |

### `python` feature + cvxpy (needs a venv with cvxpy)

| Example | Command | Plot |
|---|---|---|
| `mean_variance_strategy` | `cargo run --example mean_variance_strategy --features python -- <common> --window 20` | `python examples/plot_strategy.py target/mean_variance_strategy.csv` |
| `benchmark_relative_strategy` | `cargo run --example benchmark_relative_strategy --features python -- <common> --window 20` | `python examples/plot_strategy.py target/benchmark_relative_strategy.csv` |
| `covariance_gmv` | `cargo run --example covariance_gmv --features python -- <common> --window 20` | `python examples/plot_strategy.py target/covariance_gmv.csv` |

### Parallelism

Most `python` examples take `--threads N` (default `0` = serial). The `flowgraph`
`Pool` overlaps operators whose work **releases the GIL** — NumPy / BLAS and the
cvxpy solve — so independent per-config solves (e.g. the risk-aversion sweep in
`mean_variance_strategy`) run truly in parallel even on a GIL interpreter.
Pure-Python glue still serializes under the GIL; that, plus the largely-serial
data load, is why end-to-end speedup is well under `N×`.

> **Required when `--threads N > 0`: disable OpenBLAS's internal threading**
> (`OPENBLAS_NUM_THREADS=1`). `env.ps1` sets it; on Unix `export` it before the
> interpreter loads NumPy. **OpenBLAS is not thread-safe with its internal
> parallelism enabled** — its worker pool is created/resized lazily, so when
> several `Pool` workers drive BLAS / LAPACK at once (e.g. `covariance_gmv`'s
> covariance estimators + Markowitz solves) the concurrent pool init corrupts
> OpenBLAS's internal state and **crashes** (a segfault; `0xC0000005` on
> Windows). Setting `OPENBLAS_NUM_THREADS=1` leaves the `flowgraph` `Pool` as the
> only source of parallelism (one BLAS call per worker), and multi-threaded runs
> become bit-identical to single-threaded. Only OpenBLAS needs this — other BLAS
> backends (e.g. MKL) are thread-safe. Symptom if you forget:
> `covariance_gmv --threads 8` crashes with no output.

---

## Troubleshooting

**A "shared library not found" failure with no traceback** (on Windows: exit
`0xC0000135` / `-1073741515`; on Linux: an `ld`/`libpython*.so` load error).
`libpython` isn't loadable — either its directory isn't on the OS library path,
or the binary was built against a *different* interpreter than the one you're
providing at run time (classic case: a binary built against a free-threaded
`python3.Xt` won't accept the GIL `libpython`). Fix: set the three knobs to one
interpreter (dot-source `env.ps1` on Windows); if you previously built against a
different interpreter, rebuild (see below).

**`ModuleNotFoundError: No module named 'flowops'`.** The repo's `python/`
directory isn't on `PYTHONPATH` and `flowops` isn't installed in the venv. Either
`pip install -e .` into the venv, or add `python/` to `PYTHONPATH`.

**`ModuleNotFoundError: No module named 'cvxpy'` (or `numpy` / `scipy`).** The
venv's `site-packages` isn't on `PYTHONPATH`, or you pointed at a venv that lacks
the package (e.g. running a cvxpy example against a NumPy-only environment). Use a
venv that has the package.

**`ImportError: DLL load failed while importing _multiarray_umath` (NumPy), or
"NumPy compiled against a different Python".** ABI mismatch: the interpreter you
*linked/loaded* (`PYO3_PYTHON` + `libpython`) is a different build from the venv
whose `site-packages` is on `PYTHONPATH` (e.g. a GIL build with free-threaded
NumPy). Keep all three knobs on one venv.

**PyO3 build fails to find Python, or links the wrong one.** `PYO3_PYTHON` was
unset and PyO3 picked a stub or a different version. Set `PYO3_PYTHON` to the
venv's interpreter and rebuild.

**You changed `PYO3_PYTHON` but the binary still loads the old library.**
Changing `PYO3_PYTHON` changes PyO3's build-config hash, so the next `cargo build`
relinks automatically. If it seems stuck, force it: `cargo clean -p pyo3-ffi -p
pyo3 -p numpy` (or `cargo clean`), then rebuild.

**(Windows / PowerShell) Red `NativeCommandError` lines, run aborts early.** The
examples write progress to **stderr** (`eprintln!`). Under
`$ErrorActionPreference = 'Stop'`, PowerShell turns the first stderr line into a
*terminating* error and kills the run even though nothing failed. Leave
`$ErrorActionPreference` at its default `Continue` (`env.ps1` does not set
`Stop`); those red lines are harmless — check the real exit code.

---

## Free-threaded CPython (optional)

The engine runs unchanged on a free-threaded build (`python3.Xt`,
`Py_GIL_DISABLED`), where pure-Python operators parallelize too. To use it,
create a separate venv from a free-threaded base and point the three knobs at it.
**Caveat:** cvxpy currently has no free-threaded wheels, so the cvxpy examples
(`mean_variance_strategy`, `benchmark_relative_strategy`, `covariance_gmv`) can't
run there — only the NumPy-only ones. That is why a standard GIL interpreter is
the default.
