use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::python::py_params;

/// Configuration shared by every portfolio.
///
/// The cross-section width is absent by design: it is read from the input
/// arrays, so it cannot disagree with the wiring.
///
/// ```
/// # use tradingflow::operators::portfolio::Config;
/// // A hundred-name book out of a much wider cross-section.
/// let config = Config {
///     max_universe_size: Some(100),
///     ..Config::default()
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Whether the incoming moments are log returns and need mapping to linear
    /// ones before optimization. On by default, since that is what the
    /// predictors produce.
    pub logarithmic: bool,
    /// Upper bound on how many stocks the universe may select, which sizes the
    /// solving portfolios' warm-started problem. `None` sizes it to the whole
    /// cross-section — correct, but wasteful when the universe is a small
    /// slice of it.
    pub max_universe_size: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            logarithmic: true,
            max_universe_size: None,
        }
    }
}

impl Config {
    /// Builds the keyword arguments for a Python portfolio's `build`, applying
    /// `extra` for the portfolio's own parameters.
    ///
    /// # Panics
    ///
    /// If the parameters cannot be converted to Python values, which for the
    /// scalars involved means the interpreter is out of memory.
    pub(crate) fn params(
        &self,
        extra: impl for<'py> FnOnce(&Bound<'py, PyDict>) -> PyResult<()>,
    ) -> Option<Py<PyDict>> {
        py_params(|dict| {
            dict.set_item("logarithmic", self.logarithmic)?;
            dict.set_item("max_universe_size", self.max_universe_size)?;
            extra(dict)
        })
    }
}
