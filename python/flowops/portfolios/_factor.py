"""Low-rank-plus-diagonal (factor-model) approximation of a covariance matrix,
shared by the warm-started cvxpy portfolio leaves.

A full `Sigma` (M x M) makes the cvxpy problem un-warm-startable: putting the
covariance (or its Cholesky/LDL factor) in as a `Parameter` costs O(M^3) to
canonicalise — the DPP map for an M x M matrix parameter has shape M x M^2 (it
OOMs around M ~ 1000), while `quad_form(x, P_param)` dodges the memory but is *not*
DPP, so it re-canonicalises every solve.

Approximating, via the top-`r` eigenpairs (an SVD low-rank approximation of the
symmetric PSD `Sigma`),

    Sigma  ~=  B B^T + diag(d^2)            (B: M x r,  r << M)

drops the parameter to `r*M + M` numbers, so the risk

    x^T Sigma x  ~=  || B^T x ||^2 + sum_i d_i^2 x_i^2
                  =  sum_squares(F @ x) + sum_squares(d (.) x)        (F = B^T)

is genuinely DPP (map shape r x (r*M)) and re-solves reuse the canonical form and
the warm start.  The diagonal is set so the approximation matches `Sigma`'s
diagonal exactly, preserving per-name variances (the dropped term is the
off-diagonal residual beyond the top `r` factors).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.sparse.linalg import eigsh as _eigsh
except Exception:  # pragma: no cover - scipy always present in the cvxpy venv
    _eigsh = None


def factor_decompose(sigma: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-`rank` SVD low-rank + idiosyncratic-diagonal split of a PSD `sigma`.

    Returns `(B, d)`: `B` the `n x r_eff` loadings (`r_eff = min(rank, n)`) and `d`
    the length-`n` idiosyncratic **std-dev** (sqrt of the residual diagonal), so
    `sigma ~= B @ B.T + diag(d**2)` with the diagonal matched exactly.
    """
    n = sigma.shape[0]
    r_eff = min(int(rank), n)
    if _eigsh is not None and 1 <= r_eff < n:
        try:
            ev, V = _eigsh(sigma, k=r_eff, which="LA")  # top-r_eff (PSD => largest)
        except Exception:
            ev, V = np.linalg.eigh(sigma)
            ev, V = ev[-r_eff:], V[:, -r_eff:]
    else:
        ev, V = np.linalg.eigh(sigma)
        ev, V = ev[-r_eff:], V[:, -r_eff:]
    B = V * np.sqrt(np.maximum(ev, 0.0))  # n x r_eff
    d2 = np.maximum(np.diag(sigma) - np.einsum("ij,ij->i", B, B), 0.0)
    return B, np.sqrt(d2)


def assign_slots(slot_of: dict, active_idx, max_size: int) -> np.ndarray:
    """Stable mapping from active stocks to fixed solver slots `[0, max_size)`.

    `slot_of` (global stock index -> slot) is carried in the operator state and
    mutated in place: stocks that are still active keep their slot, departed stocks
    free theirs, and new entrants take the lowest free slots.  Keeping continuing
    names in the same slot is what lets cvxpy's cached primal+dual warm start stay
    aligned across rebalances (the variable `.value` seed is ignored on re-solves).

    Returns the slot of each active stock, in `active_idx` order.
    """
    active = [int(g) for g in active_idx]
    active_set = set(active)
    for g in [g for g in slot_of if g not in active_set]:  # free departed slots
        del slot_of[g]
    used = set(slot_of.values())
    free = (s for s in range(int(max_size)) if s not in used)
    for g in active:
        if g not in slot_of:
            slot_of[g] = next(free)
    return np.fromiter((slot_of[g] for g in active), dtype=np.intp, count=len(active))


def factor_params_at(
    sigma: np.ndarray, slots: np.ndarray, max_size: int, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose `sigma` and scatter into the fixed solver size at `slots`.

    Returns `(F, d, active)` for the persistent problem's parameters: `F` is
    `rank x max_size` (`B.T` placed at `slots`, zero elsewhere and in unused factor
    rows), `d` is `max_size` (idiosyncratic std-dev at `slots`), and `active` is the
    0/1 slot mask.  The caller scatters `mu` (and reads weights back) at the same
    `slots`.
    """
    B, d_sub = factor_decompose(sigma, rank)
    r_eff = B.shape[1]
    m, r = int(max_size), int(rank)
    F = np.zeros((r, m))
    F[:r_eff, slots] = B.T
    d = np.zeros(m)
    d[slots] = d_sub
    active = np.zeros(m)
    active[slots] = 1.0
    return F, d, active
