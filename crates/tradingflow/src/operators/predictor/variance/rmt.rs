use pyo3::types::PyDictMethods;

use super::VariancePredictor;
use crate::operators::predictor::Config;
use crate::python::py_segment_module;

/// What [`rmt`] does with the eigenvalues it judges to be noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replacement {
    /// Discards them (Rosenow et al., 2002). Leaves the filtered correlation
    /// singular on the discarded subspace, which is honest about carrying no
    /// information there but can be awkward for an optimizer that wants to
    /// invert it.
    Zero,
    /// Replaces them with their mean (Potters et al., 2005), preserving the
    /// trace and so the total variance. Keeps the matrix full rank.
    Mean,
}

/// Random matrix theory filter: keeps only the eigenvalues of the sample
/// correlation that noise alone cannot explain.
///
/// The eigenvalue spectrum of a correlation matrix estimated from pure noise
/// has a known upper edge — the Marchenko-Pastur bound `σ²(1 + N/T +
/// 2√(N/T))`, with the Laloux correction `σ² = 1 - λ₁/N` removing the market
/// mode from the null. Eigenvalues below it are indistinguishable from noise,
/// so the filter suppresses them and rebuilds the correlation from what
/// remains. `replacement` chooses how.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn rmt(config: Config, replacement: Replacement) -> impl VariancePredictor {
    let mode = match replacement {
        Replacement::Zero => "zero",
        Replacement::Mean => "mean",
    };
    py_segment_module(
        "tradingflow.predictor.variance.rmt",
        config.params(|d| d.set_item("mode", mode)),
    )
}
