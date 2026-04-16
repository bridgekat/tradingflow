"""Hierarchical-clustering covariance predictors.

Three agglomerative hierarchical-clustering estimators of the
covariance matrix, following Pantaleo et al. (2010).  All three build
a dendrogram from the sample correlation similarities, read the
*cophenetic similarity* (the similarity at which pairs first merge)
into a filtered correlation matrix, and rescale by the sample standard
deviations.  They differ only in the rule that defines the similarity
between a newly-merged cluster ``L = A ∪ B`` and another active
cluster ``F``:

- [`UPGMA`][tradingflow.operators.predictors.variance.UPGMA] — unweighted
  pair-group method with arithmetic mean:
  ``σ(L, F) = (N_A σ(A, F) + N_B σ(B, F)) / (N_A + N_B)``.
- [`WPGMA`][tradingflow.operators.predictors.variance.WPGMA] — weighted
  pair-group method with arithmetic mean:
  ``σ(L, F) = (σ(A, F) + σ(B, F)) / 2``.
- [`Hausdorff`][tradingflow.operators.predictors.variance.Hausdorff] —
  Hausdorff linkage using the original pairwise similarities:
  ``σ(L, F) = min{min_{i∈L} max_{j∈F} σ_ij, max_{i∈L} min_{j∈F} σ_ij}``.

To keep the dendrogram monotonic (a prerequisite for a positive
semi-definite filtered correlation matrix), merge similarities are
clamped to be non-increasing.  UPGMA and WPGMA are monotonic by
construction when similarities are non-negative; the clamp matters
only for the Hausdorff variant, where it plays the role of the MST
reversal-removal step described in the paper.
"""

from typing import Literal

import numpy as np

from ..variance_predictor import VariancePredictor
from ._common import correlation_from_covariance, sample_covariance


class UPGMA(VariancePredictor[np.ndarray]):
    """UPGMA hierarchical-clustering covariance estimator.

    Size-weighted arithmetic-mean linkage on the sample correlation.
    Ignores features.  See the module docstring for the estimator
    family.
    """

    def __init__(self, universe, features_series, adjusted_prices_series, **kwargs) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=lambda x, y: _hcluster_fit(y, method="upgma"),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


class WPGMA(VariancePredictor[np.ndarray]):
    """WPGMA hierarchical-clustering covariance estimator.

    Unweighted (simple-average) linkage on the sample correlation.
    Ignores features.  See the module docstring for the estimator
    family.
    """

    def __init__(self, universe, features_series, adjusted_prices_series, **kwargs) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=lambda x, y: _hcluster_fit(y, method="wpgma"),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


class Hausdorff(VariancePredictor[np.ndarray]):
    """Hausdorff-linkage hierarchical-clustering covariance estimator.

    Uses the Hausdorff-style similarity
    ``min{min_i max_j, max_i min_j}`` over the original pairwise
    correlations.  Reversals in the resulting dendrogram are removed
    by clamping each merge similarity to the minimum of the previous
    merges.  Ignores features.
    """

    def __init__(self, universe, features_series, adjusted_prices_series, **kwargs) -> None:
        super().__init__(
            universe,
            features_series,
            adjusted_prices_series,
            fit_fn=lambda x, y: _hcluster_fit(y, method="hausdorff"),
            predict_fn=lambda state, x, params: params,
            **kwargs,
        )


def _hcluster_fit(y: np.ndarray, *, method: Literal["upgma", "wpgma", "hausdorff"]) -> np.ndarray:
    S, _, _ = sample_covariance(y)
    C, stds = correlation_from_covariance(S)

    coph = _cophenetic_similarity(C, method=method)
    return coph * np.outer(stds, stds)


def _cophenetic_similarity(C: np.ndarray, *, method: str) -> np.ndarray:
    """Build the cophenetic-similarity matrix of an agglomerative clustering.

    For each method the algorithm iteratively merges the two active
    clusters with the highest similarity, records the merge similarity
    between all cross-cluster element pairs, and updates the similarity
    between the new cluster and every remaining cluster using the
    method's linkage rule.

    Parameters
    ----------
    C
        ``(N, N)`` similarity matrix (typically the sample correlation)
        with ``C[i, i] = 1``.
    method
        One of ``"upgma"``, ``"wpgma"``, ``"hausdorff"``.

    Returns
    -------
    np.ndarray
        ``(N, N)`` cophenetic-similarity matrix with unit diagonal.
    """
    N = C.shape[0]
    members: dict[int, list[int]] = {i: [i] for i in range(N)}
    sizes: dict[int, int] = {i: 1 for i in range(N)}

    # Pairwise similarities keyed by ordered (a, b) with a < b.
    sim: dict[tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            sim[(i, j)] = float(C[i, j])

    active: set[int] = set(range(N))
    next_id = N
    coph = np.eye(N, dtype=np.float64)
    prev_merge_sim = np.inf

    while len(active) > 1:
        # Most-similar active pair.
        best_key = max(sim, key=sim.__getitem__)
        a, b = best_key
        merge_sim = sim[best_key]

        # Enforce dendrogram monotonicity (reversal removal).
        merge_sim = min(merge_sim, prev_merge_sim)
        prev_merge_sim = merge_sim

        # Record cophenetic similarity for every cross-cluster pair.
        a_members = members[a]
        b_members = members[b]
        for i in a_members:
            for j in b_members:
                coph[i, j] = merge_sim
                coph[j, i] = merge_sim

        # Create the merged cluster.
        new_id = next_id
        next_id += 1
        members[new_id] = a_members + b_members
        sizes[new_id] = sizes[a] + sizes[b]

        # Update similarity with every remaining active cluster.
        for F in active:
            if F == a or F == b:
                continue
            if method == "upgma":
                s_aF = sim[(min(a, F), max(a, F))]
                s_bF = sim[(min(b, F), max(b, F))]
                new_sim = (sizes[a] * s_aF + sizes[b] * s_bF) / (sizes[a] + sizes[b])
            elif method == "wpgma":
                s_aF = sim[(min(a, F), max(a, F))]
                s_bF = sim[(min(b, F), max(b, F))]
                new_sim = 0.5 * (s_aF + s_bF)
            elif method == "hausdorff":
                # Paper's formula uses the ORIGINAL pairwise similarities
                # from C — not the running cluster-cluster similarities.
                sub = C[np.ix_(members[new_id], members[F])]
                term1 = sub.max(axis=1).min()  # min_i max_j
                term2 = sub.min(axis=1).max()  # max_i min_j
                new_sim = float(min(term1, term2))
            else:
                raise ValueError(f"unknown method {method!r}")

            sim[(min(new_id, F), max(new_id, F))] = new_sim

        # Retire the merged clusters.
        active.remove(a)
        active.remove(b)
        active.add(new_id)
        for key in list(sim.keys()):
            if a in key or b in key:
                del sim[key]

    return coph
