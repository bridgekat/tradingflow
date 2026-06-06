"""Parameter-cached Markowitz (``Mode.MIN_MEAN_VARIANCE``), variable active set.

Same long-only mean-variance program as :mod:`markowitz`, but the cvxpy
**canonicalization is built once** with ``mu``, the covariance Cholesky factor
``L``, and an ``active`` mask as DPP ``Parameter`` s (verified ``is_dpp``).  Each
``compute`` only assigns the parameter values and calls ``solve()`` — so the
per-rebalance cost is the Cholesky (LAPACK, GIL-released) plus the SCS solve (C,
GIL-released), with no Python-side re-canonicalization holding the GIL.  This is
what lets the flow ``Pool`` overlap independent per-delta solves (default
:mod:`markowitz` rebuilds the ``cp.Problem`` every call, so its canonicalization
serialises on the GIL and caps parallel speedup at ~1.6x).

**Variable active set.**  cvxpy ``Parameter`` s have fixed dimensions, but the
number of *in-universe* stocks handed to the portfolio varies per rebalance —
only the upper bound ``universe_size`` is known ahead of time.  So the program
is sized once to ``U = universe_size`` and the live ``k <= U`` active stocks are
packed into the first ``k`` slots: ``mu``/``L`` carry the active block (padding
zeros), and the ``active`` mask (1 for the first ``k`` slots, 0 otherwise)
forces the padding weights to zero (``x <= active`` with ``x >= 0``).  The
solved weights are scattered back to the full ``num_stocks`` vector.

Inputs (in order): ``(universe, predicted_returns, covariance)`` — matching
:class:`markowitz.Markowitz`.  Implements ``Mode.MIN_MEAN_VARIANCE`` and
``long_only``.

NOTE: integrating this into the production :class:`MeanVariancePortfolio` base
needs the base to pass ``universe_size`` (the upper bound) through to the
``positions_fn`` — today the base masks to the (variable) active set and hands
the subclass only the ``k``-sized ``mu``/``Sigma``, so a subclass cannot size a
cached program.  This standalone operator takes ``universe_size`` directly.
"""

import numpy as np


class MarkowitzCached:
    def __init__(self, *, num_stocks, bound, long_only=True, universe_size=None, **kwargs):
        self.num_stocks = int(num_stocks)
        self.bound = float(bound)
        self.long_only = bool(long_only)
        self.u = int(universe_size) if universe_size is not None else int(num_stocks)

    def init(self, inputs, timestamp):
        import cvxpy as cp

        u = self.u
        x = cp.Variable(u)
        mu = cp.Parameter(u)
        ell = cp.Parameter((u, u))  # active block of chol(Sigma) in the top-left
        active = cp.Parameter(u, nonneg=True)  # 1 for live slots, 0 for padding
        cons = [cp.sum(x) == 1]
        if self.long_only:
            cons += [x >= 0, x <= active]
        else:
            cons += [x <= active, x >= -active]
        obj = cp.Maximize(mu @ x - self.bound * cp.sum_squares(ell.T @ x))
        prob = cp.Problem(obj, cons)
        assert prob.is_dpp(), "cached Markowitz must be DPP to skip re-canonicalization"
        return {"prob": prob, "x": x, "mu": mu, "ell": ell, "active": active, "u": u}

    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        if not produced[0]:
            return False
        import cvxpy as cp

        universe_in, mu_in, sigma_in = inputs
        uni = np.asarray(universe_in.to_numpy(), dtype=float).ravel()
        mu_full = np.asarray(mu_in.to_numpy(), dtype=float).ravel()
        n = mu_full.shape[0]
        sigma_full = np.asarray(sigma_in.to_numpy(), dtype=float).reshape(n, n)

        # In-universe active set (variable size k <= U).
        mask = (uni > 0) & np.isfinite(mu_full) & np.isfinite(np.diag(sigma_full))
        idx = np.where(mask)[0]
        k = idx.size
        out = np.zeros(n)
        if k == 0:
            output.write(out)
            return True
        u = state["u"]
        if k > u:  # universe is constructed with <= U nonzero; clip defensively
            idx = idx[:u]
            k = u

        mu_sub = mu_full[idx]
        sig = sigma_full[np.ix_(idx, idx)]
        sig = 0.5 * (sig + sig.T)
        try:
            chol = np.linalg.cholesky(sig + 1e-10 * np.eye(k))
        except np.linalg.LinAlgError:
            output.write(np.full(n, np.nan))
            return True

        # Pack the k active stocks into the first k slots of the fixed-U program.
        mu_pad = np.zeros(u)
        mu_pad[:k] = mu_sub
        ell_pad = np.zeros((u, u))
        ell_pad[:k, :k] = chol
        act = np.zeros(u)
        act[:k] = 1.0
        state["mu"].value = mu_pad
        state["ell"].value = ell_pad
        state["active"].value = act

        try:
            state["prob"].solve(solver=cp.SCS)
        except cp.SolverError:
            output.write(np.full(n, np.nan))
            return True
        w = state["x"].value
        if w is None:
            output.write(np.full(n, np.nan))
            return True

        out[idx] = np.asarray(w[:k], dtype=float)
        output.write(out)
        return True


def build(**kwargs) -> MarkowitzCached:
    return MarkowitzCached(**kwargs)
