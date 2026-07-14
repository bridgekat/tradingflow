"""
admm_mnr.py
===========
Unified matrix-free / dense ADMM-MNR for factor-model portfolio optimization.

One solver, three operators.  The ADMM outer loop, the augmented-saddle
construction, MINRES+MNR, the box-slack update, and the active-set completion
are written ONCE.  What changes between dense and matrix-free is only how the
operator A (with A^T A = 2*delta*Sigma) computes A x and A^T r:

    DenseLSOperator       A is a stored matrix          A x = A @ x        O(p n)
    FactorLSOperator      A = [s F^T ; s sqrt(D)]       A x matrix-free    O(n k)
    BlockDiagLSOperator   K identical blocks of a base  per-block matvec   O(N k)

A single-account problem is just BlockDiagLSOperator with K = 1, so multi-account
needs no separate code path.  Box constraints lower <= x <= upper are handled
natively as the pair [-x ; x] (never forming a 2n x n matrix), and the equality
matrix E (small, m_e rows) is kept dense.

    objective:   min  0.5 || A x - b ||^2          (= 0.5 x^T Sigma x - mu^T x
                                                      when A^T b = mu, delta=0.5)
    subject to:  E x = f,   lower <= x <= upper

Public API
----------
    operator factory:   factor_operator(F, D, delta), dense_operator(A),
                        block_diag(base_op, K)
    solvers:            mnr_solve(A_op, b, E, f)          # equality-constrained LS
                        admm_mnr(A_op, b, E, f, ...)       # box + equality, completion
    convenience:        solve_portfolio(...)               # single account
                        solve_multi_account(...)           # K accounts
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import norm
from scipy import optimize


# ══════════════════════════════════════════════════════════════════════
#  Linear operators:  A with  A^T A = 2*delta*Sigma
# ══════════════════════════════════════════════════════════════════════

class LSOperator:
    """Least-squares operator. Subclasses implement matvec / rmatvec.

    shape = (p, n): p is the residual-block dimension, n the primal dimension.
    sigma_matvec(x) returns A^T A x = 2*delta*Sigma x; the default composes
    rmatvec(matvec(x)) but subclasses may override for efficiency.
    """
    shape: Tuple[int, int]

    def matvec(self, x: np.ndarray) -> np.ndarray:      # A x  -> (p,)
        raise NotImplementedError

    def rmatvec(self, r: np.ndarray) -> np.ndarray:     # A^T r -> (n,)
        raise NotImplementedError

    def sigma_matvec(self, x: np.ndarray) -> np.ndarray:  # A^T A x -> (n,)
        return self.rmatvec(self.matvec(x))

    def rhs_for_mu(self, mu: np.ndarray) -> np.ndarray:
        """Return b with A^T b = mu (so 0.5||Ax-b||^2 = 0.5 x^T(A^TA)x - mu^T x)."""
        raise NotImplementedError


class DenseLSOperator(LSOperator):
    """A stored as a dense matrix. For experiments / baselines."""
    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A, dtype=float)
        self.shape = (self.A.shape[0], self.A.shape[1])

    def matvec(self, x):  return self.A @ x
    def rmatvec(self, r): return self.A.T @ r

    def rhs_for_mu(self, mu):
        b, *_ = np.linalg.lstsq(self.A.T, mu, rcond=None)
        return b


class FactorLSOperator(LSOperator):
    """A = [sqrt(2d) F^T ; sqrt(2d) sqrt(D)], matrix-free O(nk).

    A^T A = 2d (F F^T + diag(D)) = 2d Sigma.  Residual dim p = k + n.
    """
    def __init__(self, F: np.ndarray, D: np.ndarray, delta: float = 1.0):
        self.F = np.asarray(F, dtype=float)
        self.n, self.k = self.F.shape
        self.sqrtD = np.sqrt(np.maximum(np.asarray(D, dtype=float), 1e-14))
        self.s = math.sqrt(2.0 * delta)
        self.shape = (self.k + self.n, self.n)

    def matvec(self, x):
        return np.concatenate([self.s * (self.F.T @ x), self.s * (self.sqrtD * x)])

    def rmatvec(self, r):
        return self.s * (self.F @ r[:self.k] + self.sqrtD * r[self.k:])

    def sigma_matvec(self, x):
        # 2d (F (F^T x) + D x)
        return (self.s ** 2) * (self.F @ (self.F.T @ x) + (self.sqrtD ** 2) * x)

    def rhs_for_mu(self, mu):
        b = np.zeros(self.k + self.n)
        b[self.k:] = mu / (self.s * self.sqrtD)   # A^T b = s*sqrtD*b[k:] = mu
        return b


class BlockDiagLSOperator(LSOperator):
    """K identical blocks of a base operator. Stacked x = [x_0|...|x_{K-1}].

    Used for multi-account problems sharing the same factor model.
    Single account is simply K = 1.
    """
    def __init__(self, base: LSOperator, K: int):
        self.base = base
        self.K = K
        self.pb, self.nb = base.shape
        self.shape = (self.pb * K, self.nb * K)

    def matvec(self, x):
        out = np.empty(self.pb * self.K)
        for j in range(self.K):
            out[j*self.pb:(j+1)*self.pb] = self.base.matvec(x[j*self.nb:(j+1)*self.nb])
        return out

    def rmatvec(self, r):
        out = np.empty(self.nb * self.K)
        for j in range(self.K):
            out[j*self.nb:(j+1)*self.nb] = self.base.rmatvec(r[j*self.pb:(j+1)*self.pb])
        return out

    def sigma_matvec(self, x):
        out = np.empty(self.nb * self.K)
        for j in range(self.K):
            out[j*self.nb:(j+1)*self.nb] = self.base.sigma_matvec(x[j*self.nb:(j+1)*self.nb])
        return out

    def rhs_for_mu(self, mu):
        out = np.empty(self.pb * self.K)
        for j in range(self.K):
            out[j*self.pb:(j+1)*self.pb] = self.base.rhs_for_mu(mu[j*self.nb:(j+1)*self.nb])
        return out


# operator factory helpers --------------------------------------------------

def factor_operator(F, D, delta=1.0):
    return FactorLSOperator(F, D, delta)

def dense_operator(A):
    return DenseLSOperator(A)

def dense_from_covariance(Sigma, delta=1.0, tol=1e-12):
    """Build a dense operator A with A^T A = 2*delta*Sigma via eigen sqrt."""
    S = (np.asarray(Sigma) + np.asarray(Sigma).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    keep = vals > tol * max(1.0, float(vals.max(initial=0.0)))
    A = np.sqrt(2.0 * delta * vals[keep])[:, None] * vecs[:, keep].T
    return DenseLSOperator(A)

def block_diag(base_op, K):
    return BlockDiagLSOperator(base_op, K)


# ══════════════════════════════════════════════════════════════════════
#  MINRES + MNR kernel  (operator is a plain callable: the saddle matvec)
# ══════════════════════════════════════════════════════════════════════

def row_project_rhs(E: np.ndarray, f: np.ndarray, rtol: float = 1e-12) -> np.ndarray:
    if E.shape[0] == 0:
        return np.zeros(0)
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    rank = int(np.sum(s > rtol * max(1.0, float(s[0])))) if s.size else 0
    if rank == 0:
        return np.zeros_like(f)
    Q = U[:, :rank]
    return Q @ (Q.T @ f)


def _sym_givens(a: float, b: float) -> Tuple[float, float, float]:
    a = float(a); b = float(b)
    if b == 0.0:
        return (1.0 if a == 0.0 else float(np.sign(a))), 0.0, abs(a)
    if a == 0.0:
        return 0.0, float(np.sign(b)), abs(b)
    if abs(b) > abs(a):
        t = a / b; s = float(np.sign(b)) / math.sqrt(1.0 + t*t); return s*t, s, b/s
    t = b / a; c = float(np.sign(a)) / math.sqrt(1.0 + t*t); return c, c*t, a/c


@dataclass
class MinresResult:
    raw: np.ndarray; mnr: np.ndarray; residual: np.ndarray
    kmv: int; iters: int; flag: int


def minres_mnr(op: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray,
               rtol: float = 1e-7, maxit: Optional[int] = None,
               reorth: Optional[bool] = None) -> MinresResult:
    """MINRES with minimum-norm refinement. `op` maps w -> K w (symmetric)."""
    n = rhs.shape[0]
    if maxit is None: maxit = 2 * n
    if reorth is None: reorth = n <= 900
    r2 = rhs.copy(); beta1 = float(norm(r2))
    if beta1 == 0.0:
        z = np.zeros(n); return MinresResult(z, z, z, 0, 0, 9)
    Krhs = op(rhs); kmv = 1; Kn = float(norm(Krhs))
    if Kn == 0.0:
        z = np.zeros(n); return MinresResult(z, z, rhs.copy(), kmv, 0, 9)
    V = np.zeros((n, min(maxit + 1, n + 1))) if reorth else None
    beta = 0.0; c_ = -1.0; s_ = 0.0; dbar = 0.0; eps_ = 0.0
    phi = beta1; bn = beta1
    r1 = np.zeros(n); r3 = r2.copy(); vn = r3 / bn
    if reorth: V[:, 0] = vn
    x = np.zeros(n); w = np.zeros(n); wp = np.zeros(n); res = rhs.copy()
    px = x.copy(); pr = res.copy(); pp = phi; flag = -2; it = 0
    while flag == -2:
        bp = beta; beta = bn; v = vn.copy()
        r3 = op(v); kmv += 1; it += 1; al = float(r3 @ v)
        if it > 1: r3 -= r1 * beta / bp
        r3 -= r2 * al / beta
        if reorth and it < V.shape[1]: r3 -= V[:, :it] @ (V[:, :it].T @ r3)
        r1 = r2.copy(); r2 = r3.copy(); bn = float(norm(r3))
        if bn > 1e-14:
            vn = r3 / bn
            if reorth and it < V.shape[1]: V[:, it] = vn
        else: vn = np.zeros(n)
        od = dbar; dl = c_*od + s_*al; ep = eps_; gb = s_*od - c_*al
        eps_ = s_*bn; dbar = -c_*bn
        pp = phi; pr = res.copy(); px = x.copy()
        c_, s_, gm = _sym_givens(gb, bn)
        if gm > 1e-14:
            tau_ = c_*phi; phi = s_*phi
            wo = wp.copy(); wp = w.copy()
            w = (v - ep*wo - dl*wp) / gm; x += tau_*w
            res = (s_**2)*pr - phi*c_*vn
        else: res = pr.copy(); x = px.copy(); flag = 2
        Ksp = pp * math.sqrt(gb**2 + dbar**2)
        if Ksp / Kn <= rtol: flag = 5
        if it >= maxit: flag = 4
    raw = px.copy(); r_ = pr.copy(); rr = float(r_ @ r_)
    mnr = raw - float(r_ @ raw) / rr * r_ if rr > 1e-30 else raw.copy()
    return MinresResult(raw, mnr, r_, kmv, it, flag)


# ══════════════════════════════════════════════════════════════════════
#  Equality-constrained least squares:  min 0.5||A x - b||^2  s.t. E x = f
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MnrResult:
    x_raw: np.ndarray; x_mnr: np.ndarray
    lambda_raw: np.ndarray; lambda_mnr: np.ndarray
    residual: np.ndarray; kmv: int; iters: int; flag: int


def mnr_solve(A_op: LSOperator, b: np.ndarray, E: np.ndarray, f: np.ndarray,
              rtol: float = 1e-8, maxit: Optional[int] = None) -> MnrResult:
    """Solve the equality-constrained LS via MINRES+MNR on the saddle system

        [ I    A    0 ] [r  ]   [b ]
        [ A^T  0  -E^T] [x  ] = [0 ]
        [ 0   -E    0 ] [lam]   [-f]

    Works for any LSOperator (dense / factor / block-diagonal)."""
    p, n = A_op.shape
    m = E.shape[0]
    N = p + n + m

    def op(w):
        r = w[:p]; x = w[p:p+n]; lam = w[p+n:]
        out = np.empty(N)
        out[:p] = r + A_op.matvec(x)
        ox = A_op.rmatvec(r)
        if m:
            ox = ox - E.T @ lam
            out[p+n:] = -(E @ x)
        out[p:p+n] = ox
        return out

    rhs = np.zeros(N); rhs[:p] = b
    if m: rhs[p+n:] = -f
    res = minres_mnr(op, rhs, rtol=rtol, maxit=maxit)
    return MnrResult(res.raw[p:p+n].copy(), res.mnr[p:p+n].copy(),
                     res.raw[p+n:].copy(), res.mnr[p+n:].copy(),
                     res.residual.copy(), res.kmv, res.iters, res.flag)


# ══════════════════════════════════════════════════════════════════════
#  Active-set completion + KKT certificate  (operator-based)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CompletionInfo:
    attempted: bool = False; accepted: bool = False; iteration: int = -1
    active_size: int = 0; inactive_violation: float = np.inf
    eq_proj_residual: float = np.inf; stationarity: float = np.inf
    min_active_multiplier: float = np.nan; multiplier_feasible: bool = False
    kmv: int = 0


def completion_certificate(grad: np.ndarray, E: np.ndarray, G_A: np.ndarray,
                           tol: float = 1e-7):
    """Solve a small LP for KKT multipliers: E^T lam + G_A^T nu + grad = 0, nu>=0."""
    n = grad.shape[0]; me = E.shape[0]; ma = G_A.shape[0]
    if ma == 0:
        if me == 0:
            stat = float(norm(grad, ord=np.inf)); return stat <= tol, stat, np.nan
        lam, *_ = np.linalg.lstsq(E.T, -grad, rcond=None)
        stat = float(norm(E.T @ lam + grad, ord=np.inf)); return stat <= tol, stat, np.nan
    Aeq = np.hstack([E.T if me else np.zeros((n, 0)), G_A.T])
    beq = -grad; c = np.zeros(me + ma)
    bounds = [(None, None)] * me + [(0.0, None)] * ma
    lp = optimize.linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not lp.success:
        return False, np.inf, np.nan
    stat = float(norm(Aeq @ lp.x - beq, ord=np.inf))
    nu = lp.x[me:]; min_nu = float(np.min(nu)) if ma else np.nan
    return stat <= tol and min_nu >= -tol, stat, min_nu


def active_set_completion(A_op: LSOperator, b: np.ndarray, E: np.ndarray,
                          f: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                          active_lo: np.ndarray, active_hi: np.ndarray, *,
                          rtol: float = 1e-10, feasibility_tol: float = 2e-6,
                          cert_tol: float = 1e-7):
    """Fix active box vars to their bounds (as equality rows appended to E),
    solve the equality-constrained face by mnr_solve, and verify KKT by LP.
    Works for any operator since it only augments E (no operator restriction)."""
    n = A_op.shape[1]
    me = E.shape[0]
    # box rows as unit vectors e_i for active lower/upper
    rows = []
    rhs_rows = []
    g_rows = []  # G_A rows for the inequality multipliers (lower: -e_i; upper: +e_i)
    for i in active_hi:
        e = np.zeros(n); e[i] = 1.0
        rows.append(e); rhs_rows.append(float(upper[i])); g_rows.append(e)
    for i in active_lo:
        e = np.zeros(n); e[i] = 1.0
        rows.append(e); rhs_rows.append(float(lower[i])); g_rows.append(-e)
    if rows:
        E_aug = np.vstack([E] + rows) if me else np.vstack(rows)
        f_aug = np.concatenate([f, np.asarray(rhs_rows)]) if me else np.asarray(rhs_rows)
        G_A = np.vstack(g_rows)
    else:
        E_aug = E.copy(); f_aug = f.copy(); G_A = np.zeros((0, n))

    comp = CompletionInfo(attempted=True,
                          active_size=int(active_hi.size + active_lo.size))
    sol = mnr_solve(A_op, b, E_aug, f_aug, rtol=rtol)
    x = sol.x_raw; comp.kmv = sol.kmv

    fR = row_project_rhs(E, f) if me else np.zeros(0)
    comp.eq_proj_residual = float(norm(E @ x - fR, ord=np.inf)) if me else 0.0
    viol = -np.inf
    if active_hi.size or active_lo.size or True:
        # inactive box feasibility
        lo_v = float(np.max(lower - x)) if n else -np.inf
        hi_v = float(np.max(x - upper)) if n else -np.inf
        viol = max(lo_v, hi_v)
    comp.inactive_violation = viol

    grad = A_op.sigma_matvec(x) - (A_op.rmatvec(b))  # A^T A x - A^T b = grad of 0.5||Ax-b||^2
    ok, stat, min_nu = completion_certificate(grad, E, G_A, tol=cert_tol)
    comp.stationarity = stat; comp.min_active_multiplier = min_nu
    comp.multiplier_feasible = ok
    comp.accepted = (comp.eq_proj_residual <= feasibility_tol
                     and comp.inactive_violation <= feasibility_tol and ok)
    return (x if comp.accepted else None), comp


# ══════════════════════════════════════════════════════════════════════
#  Main ADMM-MNR  (box + equality, native box handling, completion)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AdmmResult:
    x: np.ndarray
    lam: np.ndarray = field(default_factory=lambda: np.zeros(0))
    history: List[Dict[str, float]] = field(default_factory=list)
    total_kmv: int = 0
    outer_iters: int = 0
    status: str = "maxit"
    completion: CompletionInfo = field(default_factory=CompletionInfo)
    time: float = 0.0


def admm_mnr(A_op: LSOperator, b: np.ndarray, E: Optional[np.ndarray] = None,
             f: Optional[np.ndarray] = None, *,
             lower=0.0, upper=np.inf, tau: float = 2.0, mu_prox: float = 1e-2,
             outer_tol: float = 1e-3, max_outer: int = 400,
             inner_rtol0: float = 1e-4, inner_power: float = 1.25,
             inner_floor: float = 1e-8, completion: bool = True,
             stable_q: int = 5, min_completion_iter: int = 10,
             active_tol: float = 1e-7, cert_tol: float = 1e-7,
             feasibility_tol: float = 2e-6, verbose: bool = False) -> AdmmResult:
    """
    min 0.5||A x - b||^2  s.t.  E x = f,  lower <= x <= upper.

    A_op is any LSOperator. Box constraints handled natively as [-x; x]
    (no 2n x n matrix). Completion appends active box rows to E.
    """
    t0 = time.time()
    p, n = A_op.shape
    lower = np.full(n, float(lower)) if np.isscalar(lower) else np.asarray(lower, float)
    upper = np.full(n, float(upper)) if np.isscalar(upper) else np.asarray(upper, float)
    sqrt_tau = math.sqrt(tau); sqrt_mu = math.sqrt(mu_prox)
    if E is None:
        E = np.ones((1, n)); f = np.array([1.0])
    me = E.shape[0]

    # box slack: g(x) = [-x; x] <= h = [-lower; upper]
    h = np.concatenate([-lower, upper])
    mg = 2 * n
    p_tot = p + mg + n           # A-residual + box-residual + proximal
    N = p_tot + n + me

    def saddle(w):
        r_A = w[:p]; r_G = w[p:p+mg]; r_I = w[p+mg:p_tot]
        x = w[p_tot:p_tot+n]; lam = w[p_tot+n:]
        out = np.empty(N)
        out[:p] = r_A + A_op.matvec(x)
        out[p:p+n] = r_G[:n] - sqrt_tau * x          # -I block
        out[p+n:p+mg] = r_G[n:] + sqrt_tau * x       # +I block
        out[p+mg:p_tot] = r_I + sqrt_mu * x
        ox = A_op.rmatvec(r_A) + sqrt_tau * (-r_G[:n] + r_G[n:]) + sqrt_mu * r_I
        if me:
            ox = ox - E.T @ lam
            out[p_tot+n:] = -(E @ x)
        out[p_tot:p_tot+n] = ox
        return out

    b_inner = b
    x = np.zeros(n)
    Gx = np.concatenate([-x, x])
    z = np.minimum(Gx, h); u = np.zeros(mg)
    fR = row_project_rhs(E, f) if me else np.zeros(0)
    hist: List[Dict[str, float]] = []
    total_kmv = 0; prev_act = None; stable = 0
    status = "maxit"; lam_out = np.zeros(me); comp = CompletionInfo()

    for it in range(max_outer):
        eta = max(inner_floor, inner_rtol0 / ((it + 1.0) ** inner_power))
        rhs = np.zeros(N)
        rhs[:p] = b_inner
        rhs[p:p+mg] = sqrt_tau * (z - u)
        rhs[p+mg:p_tot] = sqrt_mu * x
        if me: rhs[p_tot+n:] = -f
        res = minres_mnr(saddle, rhs, rtol=eta, maxit=min(2*N, 8000))
        total_kmv += res.kmv
        x_new = res.raw[p_tot:p_tot+n].copy()
        lam_out = res.mnr[p_tot+n:].copy() if me else np.zeros(0)

        Gx = np.concatenate([-x_new, x_new])
        y = Gx + u; z_new = np.minimum(h, y); u = u + Gx - z_new

        rp = float(norm(Gx - z_new)); rd = float(tau * norm(z_new - z))
        eq = float(norm(E @ x_new - fR)) if me else 0.0
        Ax = A_op.matvec(x_new)
        obj = 0.5 * float(norm(Ax - b) ** 2)
        hist.append(dict(iter=float(it+1), obj=obj, primal=rp, dual=rd,
                         eq_proj=eq, kmv=float(total_kmv)))

        hi = np.flatnonzero(x_new >= upper - active_tol)
        lo = np.flatnonzero(x_new <= lower + active_tol)
        act = (tuple(hi.tolist()), tuple(lo.tolist()))
        stable = stable + 1 if act == prev_act else 1
        prev_act = act
        z = z_new; x = x_new

        if verbose and it % 25 == 0:
            print(f"  it={it:4d} obj={obj:.6f} rp={rp:.2e} rd={rd:.2e} "
                  f"inner={res.kmv} hi={hi.size} lo={lo.size}")

        if completion and it+1 >= min_completion_iter and stable >= stable_q:
            xc, comp = active_set_completion(A_op, b, E, f, lower, upper, lo, hi,
                                             feasibility_tol=feasibility_tol,
                                             cert_tol=cert_tol)
            comp.iteration = it + 1; total_kmv += comp.kmv
            if xc is not None:
                x = xc; status = "completed"; break

        if rp <= outer_tol and rd <= outer_tol and eq <= 10*outer_tol:
            status = "converged"; break

    x = np.minimum(np.maximum(x, lower), upper)
    return AdmmResult(x=x, lam=lam_out, history=hist, total_kmv=total_kmv,
                      outer_iters=len(hist), status=status, completion=comp,
                      time=time.time()-t0)


# ══════════════════════════════════════════════════════════════════════
#  Convenience builders
# ══════════════════════════════════════════════════════════════════════

def _sector_matrix(sector_ids, n):
    ns = int(sector_ids.max()) + 1
    S = np.zeros((ns, n))
    for j in range(ns):
        S[j, sector_ids == j] = 1.0
    return S, ns


def solve_portfolio(*, factor_loadings=None, specific_var=None, covariance=None,
                    expected_returns=None, target_weights=None, delta=1.0,
                    budget=1.0, sector_ids=None, sector_targets=None,
                    redundant_budget=True, cap=None, lower=0.0,
                    dense=False, tau=2.0, mu_prox=1e-2, outer_tol=1e-3,
                    max_outer=400, completion=True):
    """Single-account portfolio. Builds the operator + E + box and calls admm_mnr.

    Provide factor_loadings (+ specific_var) for matrix-free, or covariance for
    dense.  Objective: min 0.5 x^T Sigma x - mu^T x with Sigma scaled by delta,
    where mu = expected_returns; or tracking 0.5(x-w)^T Sigma (x-w) via
    target_weights.
    """
    if factor_loadings is not None:
        F = np.asarray(factor_loadings, float); n = F.shape[0]
        D = np.full(n, 1e-4) if specific_var is None else np.asarray(specific_var, float)
        A_op = (DenseLSOperator(np.vstack([math.sqrt(2*delta)*F.T,
                                           math.sqrt(2*delta)*np.diag(np.sqrt(D))]))
                if dense else FactorLSOperator(F, D, delta))
        sig = lambda v: 2*delta*(F @ (F.T @ v) + D * v)
    elif covariance is not None:
        Sigma = np.asarray(covariance, float); n = Sigma.shape[0]
        A_op = dense_from_covariance(Sigma, delta)
        sig = lambda v: 2*delta*(Sigma @ v)
    else:
        raise ValueError("provide factor_loadings or covariance")

    # rhs b
    if target_weights is not None:
        b = A_op.matvec(np.asarray(target_weights, float))
    elif expected_returns is not None:
        b = A_op.rhs_for_mu(np.asarray(expected_returns, float))
    else:
        b = np.zeros(A_op.shape[0])

    # equality: budget (+ sector) (+ redundant budget)
    E_rows = [np.ones(n)]; f_rows = [float(budget)]
    if sector_ids is not None:
        S, ns = _sector_matrix(np.asarray(sector_ids), n)
        if sector_targets is None:
            # neutral to a cap-weighted bench implied by target_weights or equal
            wb = np.asarray(target_weights, float) if target_weights is not None else np.full(n, budget/n)
            sector_targets = S @ wb
        for j in range(S.shape[0]):
            if sector_targets[j] > 1e-12:
                E_rows.append(S[j]); f_rows.append(float(sector_targets[j]))
    if redundant_budget:
        E_rows.append(np.ones(n)); f_rows.append(float(budget))
    E = np.vstack(E_rows); f = np.asarray(f_rows)

    up = np.inf if cap is None else cap
    res = admm_mnr(A_op, b, E, f, lower=lower, upper=up, tau=tau, mu_prox=mu_prox,
                   outer_tol=outer_tol, max_outer=max_outer, completion=completion)
    x = res.x
    metrics = dict(objective=0.5*float(x @ sig(x)) - (float(b @ A_op.matvec(x)) if False else 0.0),
                   sum_weights=float(x.sum()),
                   eq_projected=float(norm(E @ x - row_project_rhs(E, f))),
                   min_weight=float(x.min()), max_weight=float(x.max()),
                   outer_iterations=res.outer_iters, K_matvecs=res.total_kmv,
                   status=res.status, time=res.time,
                   completion_accepted=res.completion.accepted)
    return dict(weights=x, lam=res.lam, metrics=metrics, history=res.history, result=res)


def solve_multi_account(*, factor_loadings, specific_var, expected_returns,
                        budgets, delta=1.0, sector_ids=None, w_bench=None,
                        redundant_firm=True, cap=0.05, lower=0.0,
                        tau=2.0, mu_prox=1e-2, outer_tol=1e-3, max_outer=400,
                        completion=False):
    """K accounts sharing one factor model. Builds BlockDiag operator + stacked
    equalities and calls admm_mnr.  expected_returns: (K,n) or (n,) broadcast."""
    F = np.asarray(factor_loadings, float); n, k = F.shape
    D = np.asarray(specific_var, float)
    K = len(budgets)
    base = FactorLSOperator(F, D, delta)
    A_op = BlockDiagLSOperator(base, K)
    N = n * K

    mu = np.asarray(expected_returns, float)
    if mu.ndim == 1: mu = np.tile(mu, (K, 1))
    mu_flat = mu.reshape(-1)
    b = A_op.rhs_for_mu(mu_flat)

    # stacked equalities
    E_rows, f_rows = [], []
    for j in range(K):
        r = np.zeros(N); r[j*n:(j+1)*n] = 1.0
        E_rows.append(r); f_rows.append(float(budgets[j]))
    if redundant_firm:
        E_rows.append(np.ones(N)); f_rows.append(float(np.sum(budgets)))
    if sector_ids is not None and w_bench is not None:
        S, ns = _sector_matrix(np.asarray(sector_ids), n)
        bs = S @ np.asarray(w_bench, float)
        for j in range(K):
            for s in range(ns):
                if bs[s] > 1e-12:
                    r = np.zeros(N); r[j*n:(j+1)*n] = S[s]
                    E_rows.append(r); f_rows.append(float(budgets[j]*bs[s]))
    E = np.vstack(E_rows); f = np.asarray(f_rows)

    res = admm_mnr(A_op, b, E, f, lower=lower, upper=cap, tau=tau, mu_prox=mu_prox,
                   outer_tol=outer_tol, max_outer=max_outer, completion=completion)
    W = res.x.reshape(K, n)
    metrics = dict(sum_weights_per_account=[float(W[j].sum()) for j in range(K)],
                   eq_projected=float(norm(E @ res.x - row_project_rhs(E, f))),
                   outer_iterations=res.outer_iters, K_matvecs=res.total_kmv,
                   status=res.status, time=res.time)
    return dict(weights=W, lam=res.lam, metrics=metrics, history=res.history, result=res)
