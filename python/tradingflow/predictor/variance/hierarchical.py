r"""Hierarchical-clustering covariance predictors, following Pantaleo et al.
(2010).

All three build a dendrogram from the sample correlation similarities, read the
cophenetic similarity into a filtered correlation matrix, and rescale by the
sample standard deviations. They differ only in the cluster-cluster linkage
rule: `upgma` is size-weighted arithmetic-mean linkage, `wpgma` is unweighted
simple-average linkage, and `hausdorff` is Hausdorff linkage on the *original*
pairwise similarities.

Merge similarities are clamped non-increasing to keep the dendrogram monotonic,
which is what makes the filtered correlation matrix positive semi-definite.
"""

import numpy as np

from ._base import VariancePredictor, covariance_predictor
from ._common import correlation_from_covariance, sample_covariance

METHODS = ("upgma", "wpgma", "hausdorff")


def cophenetic_similarity(corr: np.ndarray, *, method: str) -> np.ndarray:
    """The cophenetic-similarity matrix of an agglomerative clustering.

    Repeatedly merges the two most similar active clusters, records the merge
    similarity for every cross-cluster element pair, and updates the similarity
    between the new cluster and each remaining one by the linkage rule.
    """
    n = corr.shape[0]
    members = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}

    # Pairwise similarities keyed by ordered (a, b) with a < b.
    sim = {(i, j): float(corr[i, j]) for i in range(n) for j in range(i + 1, n)}

    active = set(range(n))
    next_id = n
    coph = np.eye(n)
    previous = np.inf

    while len(active) > 1:
        a, b = max(sim, key=sim.__getitem__)
        # Enforce dendrogram monotonicity by removing reversals.
        merge = min(sim[(a, b)], previous)
        previous = merge

        for i in members[a]:
            for j in members[b]:
                coph[i, j] = coph[j, i] = merge

        merged = next_id
        next_id += 1
        members[merged] = members[a] + members[b]
        sizes[merged] = sizes[a] + sizes[b]

        for other in active:
            if other in (a, b):
                continue
            if method == "hausdorff":
                # The paper's formula reads the ORIGINAL pairwise similarities,
                # not the running cluster-cluster ones.
                sub = corr[np.ix_(members[merged], members[other])]
                new = float(min(sub.max(axis=1).min(), sub.min(axis=1).max()))
            else:
                s_a = sim[(min(a, other), max(a, other))]
                s_b = sim[(min(b, other), max(b, other))]
                if method == "upgma":
                    new = (sizes[a] * s_a + sizes[b] * s_b) / (sizes[a] + sizes[b])
                elif method == "wpgma":
                    new = 0.5 * (s_a + s_b)
                else:
                    raise ValueError(f"unknown linkage {method!r}")
            sim[(min(merged, other), max(merged, other))] = new

        active.difference_update((a, b))
        active.add(merged)
        for key in [k for k in sim if a in k or b in k]:
            del sim[key]

    return coph


def fit(y: np.ndarray, *, method: str) -> np.ndarray:
    """Clusters the sample correlation and rescales the cophenetic similarity
    back to a covariance."""
    s, _, _ = sample_covariance(y)
    corr, stds = correlation_from_covariance(s)
    return cophenetic_similarity(corr, method=method) * np.outer(stds, stds)


def build(*, method: str = "upgma", **kwargs) -> VariancePredictor:
    """Constructs a hierarchical-clustering covariance predictor: `method` is
    `"upgma"`, `"wpgma"` or `"hausdorff"`."""
    assert method in METHODS, f"unknown linkage {method!r}"
    return covariance_predictor(lambda y: fit(y, method=method), **kwargs)
