use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use crate::python::py_params;

/// Fitting configuration shared by every predictor.
///
/// The panel dimensions are deliberately absent: the cross-section width and
/// feature count are read from the input arrays, so they cannot disagree with
/// the wiring.
///
/// ```
/// # use tradingflow::operators::predictor::Config;
/// // Refit monthly on a two-year window of daily samples, predicting one
/// // period ahead, and skip stocks with less than a quarter of coverage.
/// let config = Config {
///     target_offset: 1,
///     refit_every: 21,
///     max_periods: Some(504),
///     min_periods: Some(63),
///     ..Config::default()
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Forward offset pairing `features[i]` with `target[i + target_offset]`.
    /// Zero fits contemporaneously, which is only meaningful when the target
    /// is already a forward return.
    pub target_offset: usize,
    /// Refit cadence in rebalances. `1` refits every rebalance; larger values
    /// reuse the last fit in between, which is how an expensive estimator
    /// stays affordable at a fast rebalance cadence.
    pub refit_every: usize,
    /// Size of the training window in sampling periods. `None` fits on every
    /// pair recorded so far and grows without bound.
    pub max_periods: Option<usize>,
    /// Minimum valid observations a stock needs within the window before it is
    /// predicted for. `None` predicts for every stock in the universe.
    pub min_periods: Option<usize>,
    /// Upper bound on the number of stocks the universe may select, checked at
    /// each rebalance. `None` leaves the universe unchecked.
    pub universe_size: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target_offset: 0,
            refit_every: 1,
            max_periods: None,
            min_periods: None,
            universe_size: None,
        }
    }
}

impl Config {
    /// Builds the keyword arguments for a Python predictor's `build`, applying
    /// `extra` for the estimator's own parameters.
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
            dict.set_item("target_offset", self.target_offset)?;
            dict.set_item("refit_every", self.refit_every)?;
            dict.set_item("max_periods", self.max_periods)?;
            dict.set_item("min_periods", self.min_periods)?;
            dict.set_item("universe_size", self.universe_size)?;
            extra(dict)
        })
    }
}
