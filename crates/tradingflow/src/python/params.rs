//! [`PyParams`] — keyword arguments for a Python operator factory.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A typed keyword argument passed to a Python operator's `build(**kwargs)`.
#[derive(Clone)]
enum Param {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
}

/// Keyword arguments for a Python operator factory. Build with the chainable
/// setters, e.g. `PyParams::new().int("num_stocks", 500).float("lam", 0.1)`.
#[derive(Clone, Default)]
pub struct PyParams(Vec<(String, Param)>);

impl PyParams {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn int(mut self, k: &str, v: i64) -> Self {
        self.0.push((k.into(), Param::Int(v)));
        self
    }
    pub fn float(mut self, k: &str, v: f64) -> Self {
        self.0.push((k.into(), Param::Float(v)));
        self
    }
    pub fn bool(mut self, k: &str, v: bool) -> Self {
        self.0.push((k.into(), Param::Bool(v)));
        self
    }
    pub fn str(mut self, k: &str, v: impl Into<String>) -> Self {
        self.0.push((k.into(), Param::Str(v.into())));
        self
    }
    pub fn ints(mut self, k: &str, v: Vec<i64>) -> Self {
        self.0.push((k.into(), Param::Ints(v)));
        self
    }
    pub fn floats(mut self, k: &str, v: Vec<f64>) -> Self {
        self.0.push((k.into(), Param::Floats(v)));
        self
    }

    pub(super) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.0 {
            match v {
                Param::Int(x) => d.set_item(k, x)?,
                Param::Float(x) => d.set_item(k, x)?,
                Param::Bool(x) => d.set_item(k, x)?,
                Param::Str(x) => d.set_item(k, x)?,
                Param::Ints(x) => d.set_item(k, x.clone())?,
                Param::Floats(x) => d.set_item(k, x.clone())?,
            }
        }
        Ok(d)
    }
}
