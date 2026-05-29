"""
test_solver_comparison.py
=========================
Head-to-head comparison of three solvers on the SAME portfolio problem,
using real A-share data:

  1. CVXPY            - tradingflow's default (markowitz.py), dense Sigma
  2. Dense-ADMM       - dense Cholesky factor A = chol(Sigma), O(n^2)/matvec
  3. Factor-ADMM-MNR  - matrix-free A = [F^T; D^{1/2}], O(nk)/matvec  (NEW)

Dense-ADMM and Factor-ADMM-MNR use the SAME admm_mnr solver; the only
difference is the operator A fed in.  This isolates the matrix-free
contribution as a single controlled variable.

Two scenarios:
  A. Single account: budget + long-only + caps        (E full rank)
  B. Multi account : K sub-portfolios, firm + account
                     budgets + per-account sectors      (E rank-deficient)

All three solve the identical objective
    min  0.5 x^T Sigma x - mu^T x
so objective values are directly comparable.

Data convention (matches python/examples/common.py):
  --data-dir  ->  symbol_list.csv  +  a_shares_history/{symbol}.daily_prices.csv

Usage (from tradingflow/python/):
  python tests/test_solver_comparison.py --data-dir ..\\data_Ashare --sizes 300 500 1000
  python tests/test_solver_comparison.py --data-dir ..\\data_Ashare --scenario single
  python tests/test_solver_comparison.py --data-dir ..\\data_Ashare --scenario multi
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locate solver + reuse data loaders from the MNR benchmark ──────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

try:
    from tradingflow.operators.portfolios.mean_variance.admm_mnr import (
        FactorLSOperator, DenseLSOperator, BlockDiagLSOperator, admm_mnr,
        mnr_solve, solve_multi_account, row_project_rhs,
    )
except Exception:
    from admm_mnr import (
        FactorLSOperator, DenseLSOperator, BlockDiagLSOperator, admm_mnr,
        mnr_solve, solve_multi_account, row_project_rhs,
    )

# reuse the loaders/factor model from the MNR benchmark if present,
# else define minimal local versions
try:
    from test_minres_vs_mnr import (
        load_universe, factor_model_pca, sector_ids_from_symbols, _zscore,
    )
except Exception:
    try:
        from benchmark_minres_vs_mnr import (
            load_universe, factor_model_pca, sector_ids_from_symbols, _zscore,
        )
    except Exception:
        raise SystemExit(
            "Could not import data loaders. Place test_minres_vs_mnr.py "
            "in the same tests/ directory."
        )

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False


# ══════════════════════════════════════════════════════════════════════
#  Problem construction (shared objective: 0.5 x^T Sigma x - mu^T x)
# ══════════════════════════════════════════════════════════════════════

def make_single_account(F, D, mu, sector_ids, wb, cap=0.02):
    """Single account: budget + long-only + caps. E is FULL RANK (budget only)."""
    n = F.shape[0]
    E = np.ones((1, n))
    f = np.array([1.0])
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.concatenate([np.zeros(n), np.full(n, cap)])
    return E, f, G, h


def make_single_account_sector(F, D, mu, sector_ids, wb, cap=0.02):
    """Single account + sector neutrality. E is RANK-DEFICIENT (redundant budget)."""
    n = F.shape[0]
    ns = int(sector_ids.max()) + 1
    S = np.zeros((ns, n))
    for j in range(ns):
        S[j, sector_ids == j] = 1.0
    bs = S @ wb
    E_rows = [np.ones(n)]
    f_rows = [1.0]
    for j in range(ns):
        if bs[j] > 1e-12:
            E_rows.append(S[j]); f_rows.append(float(bs[j]))
    E_rows.append(np.ones(n)); f_rows.append(1.0)   # redundant budget
    E = np.vstack(E_rows); f = np.asarray(f_rows)
    G = np.vstack([-np.eye(n), np.eye(n)])
    h = np.concatenate([np.zeros(n), np.full(n, cap)])
    return E, f, G, h


def make_multi_account(F, D, mu, sector_ids, wb, K=3, cap=0.05):
    """K accounts sharing the universe. Block-diagonal risk; firm + account
    budgets (redundant) + per-account sector neutrality. Returns stacked
    (Sigma_blocks via A handled by caller), E, f, G, h, and the stacked dim N=nK."""
    n = F.shape[0]
    N = n * K
    budgets = np.array([0.3, 0.4, 0.3])[:K]
    budgets = budgets / budgets.sum()
    ns = int(sector_ids.max()) + 1
    S = np.zeros((ns, n))
    for j in range(ns):
        S[j, sector_ids == j] = 1.0
    bs = S @ wb

    E_rows, f_rows = [], []
    # account budgets
    for kk in range(K):
        r = np.zeros(N); r[kk*n:(kk+1)*n] = 1.0
        E_rows.append(r); f_rows.append(float(budgets[kk]))
    # firm budget (REDUNDANT: equals sum of account budgets)
    E_rows.append(np.ones(N)); f_rows.append(1.0)
    # per-account sector neutrality + per-account sector-sum (redundant w/ acct budget)
    for kk in range(K):
        for j in range(ns):
            if bs[j] > 1e-12:
                r = np.zeros(N); r[kk*n:(kk+1)*n] = S[j]
                E_rows.append(r); f_rows.append(float(budgets[kk]*bs[j]))
    E = np.vstack(E_rows); f = np.asarray(f_rows)
    G = np.vstack([-np.eye(N), np.eye(N)])
    h = np.concatenate([np.zeros(N), np.full(N, cap)])
    return E, f, G, h, N, budgets


# ══════════════════════════════════════════════════════════════════════
#  Solver wrappers (identical objective)
# ══════════════════════════════════════════════════════════════════════

def _b_from_mu(A, mu):
    """Find b so that A^T b = mu  (=> 0.5||Ax-b||^2 = 0.5 x^T(A^TA)x - mu^T x)."""
    b, *_ = la.lstsq(A.T, mu, rcond=None)
    return b


def solve_cvxpy(Sigma, mu, E, f, G, h):
    if not HAS_CVXPY:
        return None, "no_cvxpy", 0.0, np.nan
    n = Sigma.shape[0]
    x = cp.Variable(n)
    obj = cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Sigma)) - mu @ x)
    cons = [E @ x == f, G @ x <= h]
    prob = cp.Problem(obj, cons)
    t0 = time.time()
    try:
        prob.solve(solver=cp.OSQP, max_iter=20000, verbose=False)
        dt = time.time() - t0
        if prob.status not in ("optimal", "optimal_inaccurate") or x.value is None:
            return None, prob.status, dt, np.nan
        xv = np.asarray(x.value).ravel()
        objv = 0.5 * xv @ Sigma @ xv - mu @ xv
        return xv, prob.status, dt, float(objv)
    except Exception as e:
        return None, str(e)[:40], time.time() - t0, np.nan


def solve_admm_with_A(A, Sigma, mu, E, f, G, h, *, tag=""):
    """Run admm_mnr with a dense operator A (A^T A = Sigma). Box bounds taken
    from h = [-lower; upper]. Same solver as factor path; only operator differs."""
    n = A.shape[1]
    A_op = DenseLSOperator(A)
    b = A_op.rhs_for_mu(mu)
    lower = -h[:n]; upper = h[n:]
    t0 = time.time()
    res = admm_mnr(A_op, b, E, f, lower=lower, upper=upper, tau=2.0, mu_prox=1e-2,
                   outer_tol=1e-3, max_outer=400, completion=True,
                   inner_rtol0=1e-4, inner_power=1.25, inner_floor=1e-8)
    dt = time.time() - t0
    x = res.x
    objv = 0.5 * x @ Sigma @ x - mu @ x
    return x, res, dt, float(objv)


# ══════════════════════════════════════════════════════════════════════
#  Single-account scenario
# ══════════════════════════════════════════════════════════════════════

def run_single(prices, mcap, sizes, outdir, k=10, sector_neutral=False):
    title = "SINGLE ACCOUNT" + (" + SECTOR NEUTRAL (rank-deficient E)" if sector_neutral else " (full-rank E)")
    print("\n" + "=" * 76 + f"\n  {title}\n" + "=" * 76)

    log_p = np.log(prices); returns = log_p.diff().dropna()
    common = mcap.columns.intersection(returns.columns)
    latest_cap = mcap[common].iloc[-1].dropna()

    print(f"\n  {'n':>5s} | {'CVXPY':>22s} | {'Dense-ADMM':>22s} | {'Factor-ADMM-MNR':>24s}")
    print(f"  {'':>5s} | {'time   obj      stat':>22s} | {'time   obj    Erank':>22s} | {'time   obj    avg_inner':>24s}")
    print("  " + "-" * 96)

def run_single(prices, mcap, sizes, outdir, k=10, sector_neutral=False,
               solvers=("cvxpy", "dense", "factor")):
    title = "SINGLE ACCOUNT" + (" + SECTOR NEUTRAL (rank-deficient E)" if sector_neutral else " (full-rank E)")
    print("\n" + "=" * 76 + f"\n  {title}\n" + "=" * 76)

    log_p = np.log(prices); returns = log_p.diff().dropna()
    common = mcap.columns.intersection(returns.columns)
    latest_cap = mcap[common].iloc[-1].dropna()

    print(f"\n  {'n':>5s} | {'CVXPY':>22s} | {'Dense-ADMM':>22s} | {'Factor-ADMM-MNR':>24s}")
    print(f"  {'':>5s} | {'time   obj      stat':>22s} | {'time   obj    Erank':>22s} | {'time   obj    avg_inner':>24s}")
    print("  " + "-" * 96)

    rows = []
    for n_sub in sizes:
        if n_sub > len(latest_cap):
            continue
        top = latest_cap.nlargest(n_sub).index.tolist()
        rs = returns[top].dropna()
        T, n = rs.shape
        if T < 60 or n < int(n_sub * 0.7):
            continue
        F, D = factor_model_pca(rs.values[-252:], k=k)
        Sigma = F @ F.T + np.diag(D)
        capv = mcap[top].iloc[-1].fillna(0.0).values.astype(float)
        wb = capv / capv.sum()
        mu = (_zscore(rs.values[-20:].sum(0)) * 1e-3)   # alpha signal
        sids = sector_ids_from_symbols(list(top))

        if sector_neutral:
            E, f, G, h = make_single_account_sector(F, D, mu, sids, wb)
        else:
            E, f, G, h = make_single_account(F, D, mu, sids, wb)
        rank = int(np.sum(la.svd(E, compute_uv=False) > 1e-10))

        # 1) CVXPY (dense)
        if "cvxpy" in solvers:
            x_cv, s_cv, t_cv, o_cv = solve_cvxpy(Sigma, mu, E, f, G, h)
        else:
            x_cv, s_cv, t_cv, o_cv = None, "skip", np.nan, np.nan

        # 2) Dense-ADMM (A = Cholesky factor, n x n) -- SLOW at large n
        if "dense" in solvers:
            try:
                Lc = la.cholesky(Sigma + 1e-12 * np.eye(n))
                A_dense = Lc.T                                 # A^T A = Sigma
                x_da, r_da, t_da, o_da = solve_admm_with_A(A_dense, Sigma, mu, E, f, G, h)
                da_inner = r_da.total_kmv / max(len(r_da.history), 1)
            except Exception:
                x_da, t_da, o_da, da_inner = None, 0.0, np.nan, np.nan
        else:
            x_da, t_da, o_da, da_inner = None, np.nan, np.nan, np.nan

        # 3) Factor-ADMM-MNR (TRUE matrix-free, O(nk) per matvec)
        if "factor" in solvers:
            A_fac = FactorLSOperator(F, D, delta=0.5)   # 0.5 x^T Sigma x - mu^T x
            b_fac = A_fac.rhs_for_mu(mu)
            r_fm = admm_mnr(A_fac, b_fac, E, f, lower=0.0, upper=float(h[n]),
                            tau=2.0, mu_prox=1e-2, outer_tol=5e-4, max_outer=400,
                            inner_rtol0=1e-4, inner_power=1.25, inner_floor=1e-8,
                            completion=True)
            x_fm = r_fm.x
            o_fm = 0.5 * x_fm @ Sigma @ x_fm - mu @ x_fm
            t_fm = r_fm.time
            fm_inner = r_fm.total_kmv / max(r_fm.outer_iters, 1)
        else:
            x_fm, o_fm, t_fm, fm_inner = None, np.nan, np.nan, np.nan

        def _cell(x, t, o, extra, skip):
            if "skip" == skip:
                return f"{'-':>22s}"
            if x is None:
                return f"{'FAIL':>22s}"
            return None  # filled below

        cv_str = (f"{'-':>22s}" if "cvxpy" not in solvers
                  else (f"{t_cv:6.2f} {o_cv:8.4f} {s_cv[:6]:>6s}" if x_cv is not None else f"{'FAIL':>22s}"))
        da_str = (f"{'-':>22s}" if "dense" not in solvers
                  else (f"{t_da:6.2f} {o_da:8.4f} rk{rank:>3d}" if x_da is not None else f"{'FAIL':>22s}"))
        fm_str = (f"{'-':>24s}" if "factor" not in solvers
                  else f"{t_fm:6.2f} {o_fm:8.4f} {fm_inner:6.0f}it")

        print(f"  {n:>5d} | {cv_str:>22s} | {da_str:>22s} | {fm_str:>24s}")

        agree = ""
        if x_cv is not None and x_fm is not None:
            agree = f"||x_fm-x_cv||={la.norm(x_fm-x_cv):.1e}"
        rows.append(dict(n=n, E_rank=rank,
                         cvxpy_time=t_cv, cvxpy_obj=o_cv, cvxpy_status=s_cv,
                         dense_time=t_da, dense_obj=o_da,
                         factor_time=t_fm, factor_obj=o_fm,
                         factor_avg_inner=fm_inner, agreement=agree))

    # plot timing (only solvers that ran)
    if rows:
        ns = [r["n"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 5))
        if "cvxpy" in solvers and HAS_CVXPY:
            ax.plot(ns, [r["cvxpy_time"] for r in rows], "o-", color="#9C27B0", lw=2, ms=8, label="CVXPY (dense)")
        if "dense" in solvers:
            ax.plot(ns, [r["dense_time"] for r in rows], "s-", color="#FF9800", lw=2, ms=8, label="Dense-ADMM O(n^2)")
        if "factor" in solvers:
            ax.plot(ns, [r["factor_time"] for r in rows], "^-", color="#2196F3", lw=2, ms=8, label="Factor-ADMM-MNR O(nk)")
        ax.set_xlabel("n (assets)"); ax.set_ylabel("solve time (s)")
        ax.set_title(title, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        tag = "single_sector" if sector_neutral else "single"
        p = os.path.join(outdir, f"compare_{tag}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\n  -> {p}")
    return rows


# ══════════════════════════════════════════════════════════════════════
#  Multi-account scenario
# ══════════════════════════════════════════════════════════════════════

def run_multi(prices, mcap, sizes, outdir, k=10, K=3,
              solvers=("cvxpy", "dense", "factor")):
    print("\n" + "=" * 76)
    print(f"  MULTI-ACCOUNT (K={K}, rank-deficient E with firm+account budgets)")
    print("=" * 76)

    log_p = np.log(prices); returns = log_p.diff().dropna()
    common = mcap.columns.intersection(returns.columns)
    latest_cap = mcap[common].iloc[-1].dropna()

    print(f"\n  {'n_per':>5s} {'N=nK':>6s} {'Erows':>6s} {'rank':>5s} {'red':>4s} "
          f"{'CVXPY':>16s} {'Dense-ADMM':>16s} {'Factor-ADMM-MNR':>18s}")
    print("  " + "-" * 92)

    rows = []
    for n_sub in sizes:
        if n_sub > len(latest_cap):
            continue
        top = latest_cap.nlargest(n_sub).index.tolist()
        rs = returns[top].dropna()
        T, n = rs.shape
        if T < 60 or n < int(n_sub * 0.7):
            continue
        F, D = factor_model_pca(rs.values[-252:], k=k)
        Sigma = F @ F.T + np.diag(D)
        capv = mcap[top].iloc[-1].fillna(0.0).values.astype(float)
        wb = capv / capv.sum()
        mu = (_zscore(rs.values[-20:].sum(0)) * 1e-3)
        sids = sector_ids_from_symbols(list(top))

        E, f, G, h, N, budgets = make_multi_account(F, D, mu, sids, wb, K=K)
        rank = int(np.sum(la.svd(E, compute_uv=False) > 1e-10))
        mu_full = np.tile(mu, K)

        # block-diagonal Sigma only needed for CVXPY / Dense-ADMM / objective;
        # build it only if one of those is requested (it is N x N dense)
        need_dense_sigma = ("cvxpy" in solvers) or ("dense" in solvers) or ("factor" in solvers)
        Sigma_full = None
        if need_dense_sigma:
            Sigma_full = np.zeros((N, N))
            for kk in range(K):
                Sigma_full[kk*n:(kk+1)*n, kk*n:(kk+1)*n] = Sigma

        # 1) CVXPY
        if "cvxpy" in solvers:
            x_cv, s_cv, t_cv, o_cv = solve_cvxpy(Sigma_full, mu_full, E, f, G, h)
        else:
            x_cv, s_cv, t_cv, o_cv = None, "skip", np.nan, np.nan

        # 2) Dense-ADMM (SLOW: O(N^2) per matvec; skip unless requested)
        if "dense" in solvers:
            try:
                Lc = la.cholesky(Sigma + 1e-12*np.eye(n)); A_d1 = Lc.T
                A_dense = np.zeros((n*K, N))
                for kk in range(K):
                    A_dense[kk*n:(kk+1)*n, kk*n:(kk+1)*n] = A_d1
                x_da, r_da, t_da, o_da = solve_admm_with_A(A_dense, Sigma_full, mu_full, E, f, G, h)
            except Exception:
                x_da, t_da, o_da = None, 0.0, np.nan
        else:
            x_da, t_da, o_da = None, np.nan, np.nan

        # 3) Factor-ADMM-MNR (TRUE matrix-free multi-account via BlockDiag, O(Nk))
        fm_status = "skip"
        if "factor" in solvers:
            base_op = FactorLSOperator(F, D, delta=0.5)
            A_blk = BlockDiagLSOperator(base_op, K)
            b_blk = A_blk.rhs_for_mu(mu_full)
            r_fm = admm_mnr(A_blk, b_blk, E, f, lower=0.0, upper=float(h[N]),
                            tau=2.0, mu_prox=1e-2, outer_tol=5e-4, max_outer=400,
                            inner_rtol0=1e-4, inner_power=1.25, inner_floor=1e-8,
                            completion=False)
            x_fm = r_fm.x; t_fm = r_fm.time; fm_status = r_fm.status
            o_fm = 0.5 * x_fm @ Sigma_full @ x_fm - mu_full @ x_fm
        else:
            x_fm, t_fm, o_fm = None, np.nan, np.nan

        cv_str = (f"{'-':>16s}" if "cvxpy" not in solvers
                  else (f"{t_cv:5.2f}s {o_cv:7.4f}" if x_cv is not None else f"{'FAIL':>16s}"))
        da_str = (f"{'-':>16s}" if "dense" not in solvers
                  else (f"{t_da:5.2f}s {o_da:7.4f}" if x_da is not None else f"{'FAIL':>16s}"))
        fm_str = (f"{'-':>18s}" if "factor" not in solvers
                  else f"{t_fm:5.2f}s {o_fm:7.4f}")
        print(f"  {n:>5d} {N:>6d} {E.shape[0]:>6d} {rank:>5d} {E.shape[0]-rank:>4d} "
              f"{cv_str:>16s} {da_str:>16s} {fm_str:>18s}")

        rows.append(dict(n_per=n, N=N, E_rows=E.shape[0], E_rank=rank,
                         redundant=E.shape[0]-rank,
                         cvxpy_time=t_cv, cvxpy_obj=o_cv, cvxpy_status=s_cv,
                         dense_time=t_da, dense_obj=o_da,
                         factor_time=t_fm, factor_obj=o_fm,
                         factor_status=fm_status))
    return rows


# ══════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=str, default="results_solver_comparison")
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 500, 1000])
    ap.add_argument("--scenario", choices=["single", "single_sector", "multi", "all"],
                    default="all")
    ap.add_argument("--solvers", type=str, nargs="+",
                    choices=["cvxpy", "dense", "factor"], default=None,
                    help="which solvers to run (default: all). "
                         "e.g. --solvers cvxpy factor  skips the slow Dense-ADMM")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--K", type=int, default=3, help="number of accounts (multi)")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    args = ap.parse_args(argv)

    solvers = tuple(args.solvers) if args.solvers else ("cvxpy", "dense", "factor")
    print(f"solvers: {', '.join(solvers)}")

    os.makedirs(args.outdir, exist_ok=True)
    if "cvxpy" in solvers and not HAS_CVXPY:
        print("WARNING: cvxpy not installed -> CVXPY column will show FAIL. "
              "Install with: pip install cvxpy")

    prices, mcap = load_universe(args.data_dir, args.start, args.end,
                                 max_symbols=max(args.sizes) + 200)

    all_rows = {}
    if args.scenario in ("single", "all"):
        all_rows["single"] = run_single(prices, mcap, args.sizes, args.outdir,
                                        k=args.k, sector_neutral=False, solvers=solvers)
    if args.scenario in ("single_sector", "all"):
        all_rows["single_sector"] = run_single(prices, mcap, args.sizes, args.outdir,
                                               k=args.k, sector_neutral=True, solvers=solvers)
    if args.scenario in ("multi", "all"):
        all_rows["multi"] = run_multi(prices, mcap, args.sizes, args.outdir,
                                     k=args.k, K=args.K, solvers=solvers)

    # write CSVs
    for name, rows in all_rows.items():
        if not rows:
            continue
        p = os.path.join(args.outdir, f"comparison_{name}.csv")
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"\nDone. Results in {args.outdir}/")


if __name__ == "__main__":
    main()
