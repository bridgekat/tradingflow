//! Class-based Python operator host (feature `python`).
//!
//! [`PyClassOperator`] is a graph node whose compute step runs a Python operator
//! object, mirroring the legacy `tradingflow.operator.Operator` contract, so the
//! Python-resident operator layer (predictors / portfolios / stateful
//! metrics) ports nearly verbatim:
//!
//! ```text
//! init(inputs, timestamp) -> state
//! compute(state, inputs, output, timestamp, produced) -> bool   # @staticmethod
//! ```
//!
//! `inputs` is a tuple of **views** — [`NativeArrayView`] for array edges
//! (`ArrayPort`) and [`NativeSeriesView`] for recorded-history windows
//! (`SeriesPort` edges), `None` for unit (clock) inputs. `output` is a
//! writable array view, `timestamp` is naive
//! nanoseconds (the graph's ambient event time), `produced` is a `tuple[bool, ...]`
//! parallel to `inputs`, and `state` is a Python object carried across ticks.
//!
//! The `source` is a Python *program* (statements) executed in the operator's
//! own globals (with `np`/`numpy` pre-injected) that binds the operator instance
//! to the name `__op__`.
//!
//! # Heterogeneous inputs
//!
//! The host speaks the operator library's view currency on both sides: inputs
//! are trees of port leaves (`ArrayPort<f64, N>` / `SeriesPort<f64, N>`, plus
//! `RefPort<()>` for unit clocks), runtime-length groups (`ArrayPorts` /
//! `SeriesPorts`), and tuples, so an operator's input shape is its concrete
//! [`Interface`](crate::graph::typed::Interface) type, e.g.
//! `(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)` for a
//! predictor or `ArrayPorts<f64, 1>` for an all-array operator — and the
//! output is an `ArrayPort<f64, NO>` view of the host's owned buffer. The
//! [`PyArgs`] trait walks the input type's payload tree to build the Python
//! view tuple + produced bools in one pass. (An erased enum input would have
//! to clone growing `Series` each tick — `PyArgs` reads the borrowed views
//! directly instead.)
//!
//! # Data model
//!
//! Copy-based (like the legacy bridge): each view copies the data out to a
//! fresh NumPy array on read (a strided input view is materialized
//! row-major at bind), and `output.write()` copies a NumPy array back.
//! Copies make retention safe and the cost is negligible against the NumPy/SciPy
//! math these operators run.
//!
//! # Interpreter model (single shared interpreter; easy to switch)
//!
//! The bridge embeds **one shared CPython** and enters it per `compute` via
//! PyO3's [`Python::attach`](pyo3::Python::attach). This same code runs, with
//! **no change**, on any of:
//!
//! * **Standard GIL CPython — the default.** Only one thread runs *Python* at a
//!   time, but operator work that **releases the GIL** runs truly in parallel on
//!   the pool: NumPy ufuncs/BLAS, SciPy, and convex solvers (cvxpy + CLARABEL /
//!   SCS / OSQP) all drop the GIL during their native solve. In quant operators
//!   that native/solve time *is* the cost, so solve-bound graphs parallelize;
//!   only pure-Python glue serializes. This is the default because the cvxpy
//!   solver stack has no free-threaded wheels yet.
//! * **Free-threaded CPython** (PEP 703, `python3.13t`): no GIL, so pure-Python
//!   parallelizes too. Build against it by pointing `PYO3_PYTHON` at a
//!   free-threaded venv — nothing in this module assumes one interpreter mode.
//! * **Own-GIL sub-interpreters** (PEP 684) are a possible future direction but
//!   cannot `import numpy` (NumPy declares no multiple-interpreter support), so
//!   they are not used here.
//!
//! Because operators only touch Python under `Python::attach` and never assume
//! the GIL is present or absent, switching modes is a build/runtime config change
//! (`PYO3_PYTHON` + the venv), not a code change.
//!
//! # Setup
//!
//! A venv with the operators' deps (NumPy, and SciPy/cvxpy for the optimizers):
//!
//! ```console
//! python -m venv .venv && .venv\Scripts\python -m pip install numpy scipy cvxpy
//! ```
//!
//! PyO3 links the interpreter named by `PYO3_PYTHON` at build time; the embedded
//! interpreter computes `sys.path` from the base install, so make the venv's
//! packages (and `python/`, for the `tradingflow` operator package) visible at
//! runtime via `PYTHONPATH`. Build and test with:
//!
//! ```console
//! set PYO3_PYTHON=<abs>\.venv\Scripts\python.exe
//! set PATH=<dir containing python3xx.dll>;%PATH%
//! set PYTHONPATH=<repo>\python;<abs>\.venv\Lib\site-packages
//! cargo test --features python operators::pyhost
//! ```
//!
//! For free-threaded instead, swap the venv for a `python3.13t` one (`py install
//! 3.13t-64`) and the dll dir accordingly. In production, point
//! `PYTHONPATH`/`PYTHONHOME` at the deployment environment.
//!
//! # Safety
//!
//! The view pyclasses hold a raw pointer to graph edge data — an input view's
//! backing buffer (or its owned row-major materialization) or the host's
//! output buffer — valid only for the duration of one `compute`/`init` (the
//! payload is borrowed by the engine then). Views are created fresh each call
//! and must not be retained past it. The output array view's pointer carries
//! write provenance (`&mut`); input views are read-only. `unsafe Send + Sync`
//! is sound: a node's `compute` runs on one thread at a time and views never
//! cross threads.

mod args;
mod array_view;
mod operator;
mod params;
mod series_view;

#[cfg(test)]
mod tests;

pub use args::*;
pub use array_view::*;
pub use operator::*;
pub use params::*;
pub use series_view::*;
