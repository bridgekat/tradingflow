# Running the flow-engine examples

The `.rs` files in this directory are the **Rust ports** of the A-shares examples
(originals in `python/examples/`), built on the new parallel `flow` engine
(`src/flow`, on top of `flowgraph`). They are ordinary Cargo examples:

```pwsh
cargo run --example <name> [--features pyflow] -- [args]
```

Two kinds:

* **Native examples** (`plot_daily_price`, `plot_financial_data`) are pure Rust —
  no Python, no extra setup. Just `cargo run --example plot_daily_price`.
* **`pyflow` examples** embed a CPython interpreter and call `flowops` operators
  (predictors / portfolios / traders / metrics) from Rust, with real
  NumPy / SciPy / cvxpy. These need a configured Python environment, which is
  what most of this document is about. **This is the source of the "DLL not
  found" and "flowops module not found" errors** — they are environment
  misconfiguration, not bugs in the examples.

---

## TL;DR (Windows / PowerShell)

A working GIL virtual environment already exists at `.venv` (CPython 3.13 with
`numpy`, `scipy`, `cvxpy`, `matplotlib`). To run a `pyflow` example:

```pwsh
# 1. Configure the embedded-Python env for this shell (build + runtime).
. .\examples\env.ps1

# 2. Build & run. The first pyflow build after changing interpreters relinks PyO3.
cargo run --example mean_variance_strategy --features pyflow -- --index-size 1000

# 3. Plot (use the venv's python — it has matplotlib; the bare `python` may not).
.\.venv\Scripts\python examples\plot_strategy.py target\mean_variance_strategy.csv
```

`. .\examples\env.ps1` (note the leading dot + space — *dot-sourcing*) sets the
three things the embedded interpreter needs. If you'd rather set them by hand or
understand why, read on.

---

## How the embedding works (why the env matters)

The `flow` engine has **no Python-as-host API**. Graphs are built and driven from
Rust; Python operators run *inside* an embedded CPython via PyO3
(`pyo3/auto-initialize`, feature `pyflow`). Two distinct moments need
configuration:

1. **Build time** — PyO3 links the Rust binary against one specific Python,
   chosen by the **`PYO3_PYTHON`** environment variable (an absolute path to a
   `python.exe`). That bakes in which `pythonXY.dll` the binary will load and
   which ABI it expects.

2. **Run time** — the embedded interpreter must be able to (a) *load* its DLL and
   (b) *import* the operators' Python dependencies. PyO3 does **not** activate the
   venv, so:
   * the directory containing `python313.dll` must be on **`PATH`**;
   * the `flowops` package (`python/`) and the venv's `site-packages` must be on
     **`PYTHONPATH`**.

The cardinal rule: **`PYO3_PYTHON`, the DLL on `PATH`, and the `site-packages` on
`PYTHONPATH` must all belong to the same interpreter.** Mixing a GIL DLL with
free-threaded NumPy (or vice-versa) yields cryptic import/DLL-load failures.

### The three knobs

| Variable | When | Value (for `.venv`) | If missing/wrong |
|---|---|---|---|
| `PYO3_PYTHON` | build | `<repo>\.venv\Scripts\python.exe` | links the wrong interpreter (or PyO3 falls back to the Windows Store `python` stub, which has no dev libs) |
| `PATH` (prepend) | run | `<base-prefix>` holding `python313.dll` | `STATUS_DLL_NOT_FOUND`, process exit `0xC0000135` (`-1073741515`) — *"DLL not found"* |
| `PYTHONPATH` | run | `<repo>\python;<repo>\.venv\Lib\site-packages` | `ModuleNotFoundError: flowops` / `numpy` / `cvxpy` |

For these PyManager / `pythoncore` installs the `python313.dll` lives in the
venv's **base prefix** — the `home = …` line of `.venv\pyvenv.cfg`
(e.g. `C:\Users\<you>\AppData\Local\Python\pythoncore-3.13-64`), *not* in the
venv itself. `examples\env.ps1` reads that line so you don't have to hardcode it.

### Setting the knobs by hand (equivalent to `env.ps1`)

```pwsh
$repo = (Get-Location).Path
$env:PYO3_PYTHON = "$repo\.venv\Scripts\python.exe"
$base = (Select-String -Path "$repo\.venv\pyvenv.cfg" -Pattern '^home\s*=\s*(.+)$').Matches[0].Groups[1].Value.Trim()
$env:PATH = "$base;$env:PATH"
$env:PYTHONPATH = "$repo\python;$repo\.venv\Lib\site-packages"
```

---

## One-time environment setup

The `.venv` is already present. To recreate it (or set up a fresh machine):

```pwsh
# Use the GIL CPython 3.13 managed by the `py` launcher (NOT the Store stub).
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install numpy scipy cvxpy matplotlib
#   cvxpy pulls in the solvers (clarabel / osqp / scs) automatically.
#   matplotlib is only needed for the plot_*.py scripts, not the examples.
```

Verify it (should print versions, then `flowops OK`):

```pwsh
$env:PYTHONPATH = "$((Get-Location).Path)\python"
.\.venv\Scripts\python -c "import numpy, scipy, cvxpy, matplotlib; import flowops.portfolios.mean_variance.markowitz; print('env OK')"
```

> **Note on the `python` on your `PATH`.** On this machine the bare `python` /
> `python3` resolve to the Windows Store stub (currently 3.14), which is *not*
> what the examples use. Always invoke the venv explicitly
> (`.\.venv\Scripts\python …`) for the plot scripts, and let `PYO3_PYTHON` pick
> the interpreter for builds.

---

## Getting the data

The examples read bundled A-shares market data from `python/examples/data/`
(`a_shares_history/` holds one CSV per symbol per kind, plus `symbol_list.csv`).
The crawler that produces it is vendored as a git submodule at
**`extern/a-shares-crawler`**. If the submodule is empty (fresh clone):

```pwsh
git submodule update --init extern/a-shares-crawler
```

To (re)fetch data you need an EastMoney session config — see
`extern/a-shares-crawler/README.md` for the full procedure. In short:

```pwsh
.\.venv\Scripts\python -m pip install -e extern/a-shares-crawler
.\.venv\Scripts\python -m a_shares_crawler --config config.json --data-dir python\examples\data
```

The crawler can also emit **consolidated long-format tables** (one file per kind,
all symbols, sorted by date) for a single sequential read, via
`--export-long {csv,parquet}...` (or the standalone `python -m a_shares_crawler.export`).
One or more formats may be given at once:

```pwsh
.\.venv\Scripts\python -m a_shares_crawler.export --data-dir python\examples\data --export-long csv parquet
```

This writes `python\examples\data\<kind>.{csv,parquet}`. The TradingFlow read path that
consumes these long tables (a cross-sectional `PanelSource`) is described in
[`docs/design/data-storage.md`](../docs/design/data-storage.md); until it lands,
the examples read the per-symbol CSVs.

---

## The examples

All commands assume you've dot-sourced `examples\env.ps1` first (for the
`pyflow` ones). `--index-size N` bounds the cap-weighted universe; smaller is
faster. Trim `--begin`/`--end` for quick smoke tests.

All examples now read the consolidated **long parquet** tables
(`<data-dir>/<kind>.parquet`, produced by the crawler's `--export-long parquet`)
through `ParquetPanelSource` / `ReportPanelSource`, not the per-symbol CSVs.

### Native (no `pyflow`, no Python)

| Example | Command | Plot |
|---|---|---|
| `plot_daily_price` | `cargo run --example plot_daily_price [SYMBOL]` | `python examples\plot.py target\plot_daily_price.csv` |
| `plot_financial_data` | `cargo run --example plot_financial_data [SYMBOL]` | `python examples\plot_financial_data.py target\plot_financial_data.csv` |

`ParquetPanelSource` pivots one long table into a wide `[N_symbols, K]`
cross-section per date; the per-stock pipeline is recovered by `Select`ing one
stock's row + a NaN `Filter`. The single-stock plots therefore read the whole
table (incl. the 17.5M-row `daily_prices`) to extract one stock, so a run takes
~1–2 min — the `1 → N → 1` fan-out is intentional for now.

**Semantics:** each cross-section reflects only that date's rows (absent symbols
`NaN`); the panel does **not** carry values forward or seed the window — the
"carry last value" / "NaN-fill" is the downstream `Stack` / `StackSync`'s job, as
with the old per-symbol sources. Because irregular kinds (dividends, reports) tick
on the *union* of all stocks' event dates (not every trading day), the per-stock
`Filter` drops the all-`NaN` ticks so message-passing operators like
`ForwardAdjust` see each real event once. `ReportPanelSource` aligns reports on the
**effective date** `max(report, notice)` (`use_effective_date`) so backtests don't
see a report before it was published.

### `pyflow`, NumPy only (any GIL venv with NumPy works)

| Example | Command | Plot |
|---|---|---|
| `flowops_demo` | `cargo run --example flowops_demo --features pyflow` | `plot.py target\flowops_demo.csv` |
| `plot_total_market_cap` | `cargo run --example plot_total_market_cap --features pyflow -- --index-size 1000` | `plot_total_market_cap.py target\plot_total_market_cap.csv` |
| `mean_strategy` | `cargo run --example mean_strategy --features pyflow -- --index-size 1000` | `plot_strategy.py target\mean_strategy.csv` |
| `factor_ic` | `cargo run --example factor_ic --features pyflow -- --index-size 1000` | `plot_factor_ic.py target\factor_ic.csv` |

### `pyflow` + cvxpy (needs the GIL venv with cvxpy — i.e. `.venv`)

| Example | Command | Plot |
|---|---|---|
| `mean_variance_strategy` | `cargo run --example mean_variance_strategy --features pyflow -- --index-size 1000` | `plot_strategy.py target\mean_variance_strategy.csv` |
| `benchmark_relative_strategy` | `cargo run --example benchmark_relative_strategy --features pyflow -- --index-size 1000` | `plot_strategy.py target\benchmark_relative_strategy.csv` |
| `covariance_gmv` | `cargo run --example covariance_gmv --features pyflow -- --index-size 1000` | `plot_strategy.py target\covariance_gmv.csv` |
| `bench_solves` | `cargo run --example bench_solves --features pyflow -- --n 300 --k 8 --ticks 40 --threads 8` | — (prints timings) |

> Run the plot scripts with the venv python so matplotlib is found:
> `.\.venv\Scripts\python examples\plot_strategy.py target\<file>.csv`.

### Parallelism

Most `pyflow` examples take `--threads N` (default `0` = serial). The flow `Pool`
overlaps operators whose work **releases the GIL** — NumPy/BLAS and the cvxpy
solve. So independent per-config solves (e.g. the 8 deltas in
`mean_variance_strategy`, or `bench_solves --k 8`) run truly in parallel even on
the GIL interpreter. Pure-Python glue still serializes under the GIL; that, plus
the largely-serial data load (~6000 per-stock CSVs), is why end-to-end speedup is
well under `N×`. `bench_solves` isolates the solve-parallelism from the data load.

---

## Troubleshooting

**Process exits `0xC0000135` / `-1073741515`, or "DLL not found", no traceback.**
`python313.dll` isn't loadable. Either its directory isn't on `PATH`, or the
binary was built against a *different* interpreter than the DLL you provided
(classic case: a binary built against the free-threaded `python313t` won't find
`python313t.dll`, and won't accept `python313.dll`). Fix: `. .\examples\env.ps1`;
if you previously built against another interpreter, rebuild (see below). Check
what a binary wants without running it:

```pwsh
$exe = "target\debug\examples\mean_variance_strategy.exe"
$b = [IO.File]::ReadAllBytes((Resolve-Path $exe))
(-join ($b | % { if ($_ -ge 32 -and $_ -lt 127) {[char]$_} else {"`n"} })) -split "`n" |
  Select-String 'python3\d*t?\.dll' -AllMatches | % { $_.Matches.Value } | Sort-Object -Unique
```

**`ModuleNotFoundError: No module named 'flowops'`.** `<repo>\python` isn't on
`PYTHONPATH`. Dot-source `env.ps1`, or add it.

**`ModuleNotFoundError: No module named 'cvxpy'` (or `numpy`/`scipy`).** The
venv's `site-packages` isn't on `PYTHONPATH`, or you pointed at a venv that lacks
the package (e.g. trying to run a cvxpy example against a NumPy-only env). Use
`.venv`, which has cvxpy.

**`ImportError: DLL load failed while importing _multiarray_umath` (NumPy), or
NumPy "compiled against a different Python".** ABI mismatch: the interpreter you
*linked/loaded* (`PYO3_PYTHON` + DLL on `PATH`) is a different build from the venv
whose `site-packages` is on `PYTHONPATH` — e.g. a GIL DLL with free-threaded
NumPy. Keep all three knobs on one venv.

**PyO3 build fails to find Python, or links the wrong one.** `PYO3_PYTHON` was
unset and PyO3 picked the Windows Store stub (or a different version). Set
`PYO3_PYTHON` to `.venv\Scripts\python.exe` and rebuild.

**I changed `PYO3_PYTHON` but the binary still loads the old DLL.** Changing
`PYO3_PYTHON` changes PyO3's build-config hash, so the next `cargo build`
relinks automatically. If it seems stuck, force it:
`cargo clean -p pyo3-ffi -p pyo3 -p numpy` (or `cargo clean`), then rebuild.

**PowerShell prints red `NativeCommandError` lines (e.g. after "loaded N
symbols") and the run aborts early.** The examples write progress to **stderr**
(`eprintln!`). Under `$ErrorActionPreference = 'Stop'`, PowerShell 5.1 turns the
first stderr line (especially via `2>&1`) into a *terminating* error and kills
the run — even though nothing failed. Leave `$ErrorActionPreference` at its
default `Continue`. (`env.ps1` is careful **not** to set `Stop`; if you set it
yourself, those red lines are harmless noise, not failures — check the real exit
code.)

---

## Free-threaded CPython (optional, not currently set up)

The engine runs unchanged on free-threaded CPython (`python3.13t`), where
pure-Python operators parallelize too. To use it, create a `.venv-ft` from a
`python3.13t` base and point `PYO3_PYTHON` / the DLL dir / `PYTHONPATH` at it
instead. **Caveat:** cvxpy has no free-threaded wheels yet, so the cvxpy examples
(`mean_variance_strategy`, `benchmark_relative_strategy`, `covariance_gmv`,
`bench_solves`) can't run there — only the NumPy-only ones. This is why the
default and the bundled `.venv` are the GIL interpreter.
