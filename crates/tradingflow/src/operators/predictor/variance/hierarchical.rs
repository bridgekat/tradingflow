use pyo3::types::PyDictMethods;

use super::VariancePredictor;
use crate::operators::predictor::Config;
use crate::python::py_operator_module;

/// How [`hierarchical`] measures the similarity of two clusters when deciding
/// what to merge next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linkage {
    /// Size-weighted arithmetic mean of the member similarities, so a large
    /// cluster pulls the merged similarity toward its own.
    Upgma,
    /// Unweighted mean of the two cluster similarities, giving each equal say
    /// however many members it has.
    Wpgma,
    /// Hausdorff similarity `min{minᵢmaxⱼ, maxᵢminⱼ}` over the *original*
    /// pairwise correlations rather than the running cluster similarities.
    Hausdorff,
}

/// Hierarchical-clustering filter: replaces the sample correlation with the
/// cophenetic similarity of a dendrogram fit to it.
///
/// Agglomerative clustering on the correlations produces a tree whose merge
/// heights, read back as a matrix, are a correlation with far fewer free
/// parameters — nested blocks of equicorrelated stocks rather than `N(N+1)/2`
/// independent entries. Clamping merge similarities non-increasing keeps the
/// dendrogram monotonic, which is what makes the result positive
/// semi-definite. `linkage` chooses the merge rule; see Pantaleo et al. (2010).
///
/// The clustering is `O(N³)` in the cross-section and runs on every refit, so
/// raise [`Config::refit_every`] rather than paying it at every rebalance.
///
/// See [module-level docs](super::super) for inputs and outputs.
pub fn hierarchical(config: Config, linkage: Linkage) -> impl VariancePredictor {
    let method = match linkage {
        Linkage::Upgma => "upgma",
        Linkage::Wpgma => "wpgma",
        Linkage::Hausdorff => "hausdorff",
    };
    py_operator_module(
        "tradingflow.predictor.variance.hierarchical",
        config.params(|d| d.set_item("method", method)),
    )
}
