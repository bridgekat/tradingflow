# examples/old/env.ps1 - configure the embedded-Python environment for the `python`
# examples (the Rust-driven examples in `examples/old/`).
#
# Dot-source it (note the leading "." and space) so the variables persist in
# your current shell:
#
#     . .\examples\old\env.ps1
#
# Then build/run as usual, e.g.:
#
#     cargo run --example mean_variance_strategy --features python -- --index-size 1000
#
# See examples/old/README.md for the full explanation of what each variable does.
#
# NOTE: this script deliberately does NOT set `$ErrorActionPreference = 'Stop'`.
# It is dot-sourced, so that would leak into your shell - and because the
# examples print progress to stderr (`eprintln!`), PowerShell would then wrap
# the first stderr line as a terminating NativeCommandError and abort the run
# before it does any work. Leave the preference at its default ('Continue').

# Repo root = parent of this script's directory.
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$venv = Join-Path $repo '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    throw "No .venv found at $venv. Create it first - see examples/old/README.md (`Setup`)."
}

# (1) BUILD TIME - which interpreter PyO3 links against. The GIL `.venv` (base
#     CPython 3.14) is the one cvxpy / numpy / scipy wheels were built for.
#     Changing this value forces PyO3 to reconfigure and relink on next build.
$env:PYO3_PYTHON = $venvPython

# (2) RUNTIME - the directory holding `python314.dll` must be on PATH, or the
#     embedded interpreter fails to load with STATUS_DLL_NOT_FOUND (process exit
#     0xC0000135 / -1073741515). For these PyManager/pythoncore installs the DLL
#     sits in the venv's base prefix, i.e. the `home =` line of pyvenv.cfg.
$homeMatch = Select-String -Path (Join-Path $venv 'pyvenv.cfg') -Pattern '^home\s*=\s*(.+)$'
if (-not $homeMatch) { throw "Could not read `home` from $venv\pyvenv.cfg" }
$base = $homeMatch.Matches[0].Groups[1].Value.Trim()
if ($env:PATH -notlike "*$base*") { $env:PATH = "$base;$env:PATH" }

# (3) RUNTIME - make the `tradingflow` operator package (in `python/`) and the venv's
#     site-packages importable by the embedded interpreter. PyO3 does NOT
#     auto-activate the venv, so site-packages must be added explicitly here
#     (otherwise: ModuleNotFoundError for numpy / cvxpy / tradingflow).
$env:PYTHONPATH = (Join-Path $repo 'python') + ';' + (Join-Path $venv 'Lib\site-packages')

# (4) RUNTIME - disable OpenBLAS's INTERNAL threading (`OPENBLAS_NUM_THREADS=1`).
#     OpenBLAS - the BLAS that numpy/scipy bundle (as
#     `libscipy_openblas64_*.dll`) - is NOT thread-safe with its internal
#     parallelism on: its worker pool is created/resized lazily, so when several
#     flowgraph `Pool` workers drive BLAS/LAPACK at once (e.g. `covariance_gmv`'s
#     covariance estimators + Markowitz solves) the concurrent pool init corrupts
#     OpenBLAS's internal state and crashes (a segfault; 0xC0000005 on Windows).
#     Disabling its internal parallelism leaves the flowgraph `Pool` as the only
#     source of parallelism (one BLAS call per worker); multi-threaded runs are
#     then bit-identical to single-threaded. Only OpenBLAS needs this - other
#     backends (MKL, ...) are thread-safe. Set it BEFORE the interpreter loads
#     numpy. See examples/old/README.md ("Parallelism").
$env:OPENBLAS_NUM_THREADS = '1'

Write-Host "python env configured:" -ForegroundColor Green
Write-Host "  PYO3_PYTHON = $env:PYO3_PYTHON"
Write-Host "  base (DLL)  = $base   (prepended to PATH)"
Write-Host "  PYTHONPATH  = $env:PYTHONPATH"
Write-Host "  OpenBLAS    = single-threaded (OPENBLAS_NUM_THREADS=1; OpenBLAS isn't thread-safe with internal parallelism - see README)"
