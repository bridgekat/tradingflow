use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::ffi::CString;
use std::marker::PhantomData;

use super::PyInterface;
use crate::data::Instant;
use crate::graph::{Interface, Operator};

/// Where a Python operator's definition comes from.
#[derive(Clone)]
enum Loader {
    /// Inline Python program (statements).
    Source(String),
    /// An importable module on the embedded interpreter's path.
    Module(String),
}

/// Resolves the operator instance from its [`Loader`] and keyword arguments.
fn resolve<'py>(
    py: Python<'py>,
    loader: &Loader,
    params: Option<&Py<PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let build = match loader {
        Loader::Source(src) => {
            let g = PyDict::new(py);
            let np = py.import("numpy")?;
            g.set_item("np", &np)?;
            g.set_item("numpy", &np)?;
            let code = CString::new(src.as_str())
                .map_err(|_| PyValueError::new_err("python source has interior NUL"))?;
            py.run(code.as_c_str(), Some(&g), None)?;
            g.get_item("build")?
        }
        Loader::Module(name) => py.import(name.as_str())?.getattr("build").ok(),
    };
    build
        .ok_or_else(|| PyValueError::new_err("python operator must define `build(**kwargs)`"))?
        .call((), params.map(|p| p.bind(py)))
}

/// A graph node hosting a Python operator over input interface `I` and output
/// interface `O` (both trees of array/signal ports; see [`PyInterface`]).
///
/// # The Python-side contract
///
/// The Python operator mirrors the [`Operator`] trait method-for-method:
///
/// ```python
/// class Operator:
///     type Inputs = tuple[...]
///     type Outputs = tuple[...]
///     type Context = int
///     type State = ...
///
///     def init(self, inputs: Inputs) -> State
///         ...
///
///     def reset(inputs: Inputs, state: State) -> Outputs
///         ...
///
///     def compute(inputs: Inputs, state: State, instant: Context) -> Outputs
///         ...
/// ```
///
/// `inputs` is a single tuple with one NumPy array per input leaf, in tree
/// order, and `outputs` is one value per output leaf in the same shape — a
/// bare array for a single output port, a sequence for a tuple of ports (see
/// [`PyInterface`]). `state` is an arbitrary Python object carried across
/// calls, and `instant` is the graph's ambient event time in naive
/// nanoseconds.
///
/// Like [`Operator::reset`], the Python `reset` is called once right after
/// `init` (its first return sizes the node's output buffers) and again after
/// every generation, returning quiescent outputs — signals `False`, values
/// retained or default, per the operator's semantics.
///
/// # Loading and parameters
///
/// An operator definition is loaded from an inline source, an importable
/// module, or a `.py` file; each defines a factory `build(**kwargs)`
/// returning the operator instance, called with the keyword arguments given
/// at construction as a plain Python dict (build it with PyO3, e.g.
/// [`IntoPyDict`](pyo3::types::IntoPyDict) — any Python values work).
/// A fresh instance per `build` call keeps every node's operator its own —
/// module-level state would be shared across all nodes loading the module.
///
/// # Failure
///
/// Python exceptions are printed and then panic the node, poisoning the graph
/// — the engine's no-recovery stance. Recoverable failures should be values
/// (e.g. an output signal that simply does not fire).
pub struct PyOperator<I: PyInterface, O: PyInterface> {
    loader: Loader,
    params: Option<Py<PyDict>>,
    _marker: PhantomData<fn() -> (I, O)>,
}

impl<I: PyInterface, O: PyInterface> PyOperator<I, O> {
    /// Loads from an inline Python program (handy for tests / one-offs).
    pub fn from_source(source: impl Into<String>, params: Option<Py<PyDict>>) -> Self {
        Self::new(Loader::Source(source.into()), params)
    }

    /// Loads from an importable module (on the embedded interpreter's path).
    pub fn from_module(module: impl Into<String>, params: Option<Py<PyDict>>) -> Self {
        Self::new(Loader::Module(module.into()), params)
    }

    /// Loads from a `.py` file on disk (read now).
    pub fn from_file(path: impl AsRef<std::path::Path>, params: Option<Py<PyDict>>) -> Self {
        let path = path.as_ref();
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read python operator file {}: {e}", path.display()));
        Self::from_source(source, params)
    }

    fn new(loader: Loader, params: Option<Py<PyDict>>) -> Self {
        Self {
            loader,
            params,
            _marker: PhantomData,
        }
    }
}

/// State of a [`PyOperator`] node: the Python operator instance, its Python
/// state object, and the Rust-owned buffers backing the node's outputs.
pub struct PyState<O: PyInterface> {
    instance: Py<PyAny>,
    state: Py<PyAny>,
    buffers: O::Buffers,
}

/// Builds the flat `inputs` tuple: one array per leaf, in tree order.
fn args_tuple<'py, I: PyInterface>(
    py: Python<'py>,
    values: I::Values<'_>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut args = Vec::new();
    I::to_python(py, values, &mut args)?;
    PyTuple::new(py, args)
}

/// Runs one Python `reset`/`compute` call and copies its return into the
/// node's output buffers.
fn call<'a, I: PyInterface, O: PyInterface>(
    inputs: I::Values<'a>,
    state: &'a mut PyState<O>,
    method: &str,
    instant: Option<i64>,
) -> O::Values<'a> {
    let PyState {
        instance,
        state,
        buffers,
    } = state;
    Python::attach(move |py| {
        let run = || -> PyResult<Bound<'_, PyAny>> {
            let args = args_tuple::<I>(py, inputs)?;
            let instance = instance.bind(py);
            let state = state.bind(py);
            match instant {
                Some(ns) => instance.call_method1(method, (args, state, ns)),
                None => instance.call_method1(method, (args, state)),
            }
        };
        let value = run().unwrap_or_else(|e| {
            e.print(py);
            panic!("python operator {method} failed (see traceback above)");
        });
        O::from_python(&value, buffers).unwrap_or_else(|e| {
            e.print(py);
            panic!("python operator {method} returned an incompatible value (see traceback above)");
        })
    })
}

impl<I, O> Operator for PyOperator<I, O>
where
    I: PyInterface + 'static,
    O: PyInterface + 'static,
{
    type Inputs = I;
    type Outputs = O;
    type Context = Instant;
    type State = PyState<O>;

    fn is_heavy(&self) -> bool {
        true
    }

    fn init(self, inputs: <I as Interface>::Values<'_>) -> PyState<O> {
        Python::attach(|py| {
            let run = || -> PyResult<(Py<PyAny>, Py<PyAny>)> {
                let instance = resolve(py, &self.loader, self.params.as_ref())?;
                let state = instance.call_method1("init", (args_tuple::<I>(py, inputs)?,))?;
                Ok((instance.unbind(), state.unbind()))
            };
            let (instance, state) = run().unwrap_or_else(|e| {
                e.print(py);
                panic!("python operator init failed (see traceback above)");
            });
            PyState {
                instance,
                state,
                buffers: O::Buffers::default(),
            }
        })
    }

    fn reset<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyState<O>,
    ) -> <O as Interface>::Values<'a> {
        call::<I, O>(inputs, state, "reset", None)
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut PyState<O>,
        time: &Instant,
    ) -> <O as Interface>::Values<'a> {
        call::<I, O>(inputs, state, "compute", Some(time.as_offset().as_nanos()))
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Builds the keyword arguments for a Python operator's `build`.
///
/// The operator constructors all want the same thing — attach to the
/// interpreter, fill a dict, and treat a failure as fatal — so they share this
/// rather than each repeating it.
///
/// # Panics
///
/// If the parameters cannot be converted to Python values, which for ordinary
/// scalars means the interpreter is out of memory.
pub fn py_params(
    build: impl for<'py> FnOnce(&Bound<'py, PyDict>) -> PyResult<()>,
) -> Option<Py<PyDict>> {
    Python::attach(|py| {
        let dict = PyDict::new(py);
        build(&dict).unwrap_or_else(|e| {
            e.print(py);
            panic!("cannot build python operator parameters (see traceback above)");
        });
        Some(dict.unbind())
    })
}

/// A Python operator loaded from an importable module (on the embedded
/// interpreter's path). The interfaces `I` and `O` are inferred from the
/// wiring.
pub fn py_operator_module<I: PyInterface, O: PyInterface>(
    module: impl Into<String>,
    params: Option<Py<PyDict>>,
) -> PyOperator<I, O> {
    PyOperator::from_module(module, params)
}

/// A Python operator loaded from an inline Python program defining
/// `build(**kwargs)`.
pub fn py_operator_source<I: PyInterface, O: PyInterface>(
    source: impl Into<String>,
    params: Option<Py<PyDict>>,
) -> PyOperator<I, O> {
    PyOperator::from_source(source, params)
}

/// A Python operator loaded from a plain `.py` file on disk (read now).
pub fn py_operator_file<I: PyInterface, O: PyInterface>(
    path: impl AsRef<std::path::Path>,
    params: Option<Py<PyDict>>,
) -> PyOperator<I, O> {
    PyOperator::from_file(path, params)
}
