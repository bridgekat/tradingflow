"""
benchmark_minres_vs_mnr.py
==========================
Compare MINRES (no multiplier correction) vs MINRES+MNR (with minimum-norm
refinement) on factor-model portfolio optimization, using real A-share data.

The comparison isolates the effect of MNR:
  - x-block is IDENTICAL for both (MNR does not change the primal solution)
  - lambda-block differs: MINRES gives a null-space-contaminated multiplier,
    MNR gives the minimum-norm (canonical) multiplier
  - under floating-point / data inconsistency in sector targets, ||lambda_raw||
    grows without bound while ||lambda_mnr|| stays bounded

Also reports matrix-free scaling: MINRES iteration count vs number of assets n.

Data convention (matches python/examples/common.py):
  --data-dir points to a directory containing
    symbol_list.csv
    a_shares_history/{symbol}.daily_prices.csv
    a_shares_history/{symbol}.equity_structures.csv

Usage (from tradingflow/python/):
  python benchmarks/benchmark_minres_vs_mnr.py --data-dir ../data_Ashare
  python benchmarks/benchmark_minres_vs_mnr.py --data-dir ../data_Ashare --sizes 300 500 1000

The solver module admm_mnr_core.py must be importable. This script adds the
tradingflow src directory to sys.path automatically; it also falls back to a
local copy of admm_mnr_core.py in the same directory.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Locate the solver module ──────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

try:
    # Preferred: installed as a tradingflow module
    from tradingflow.operators.portfolios.mean_variance.admm_mnr import (
        FactorLSOperator, mnr_solve, admm_mnr, row_project_rhs,
    )
except Exception:
    # Fallback: local copy next to this script
    from admm_mnr import (
        FactorLSOperator, mnr_solve, admm_mnr, row_project_rhs,
    )


# ══════════════════════════════════════════════════════════════════════
#  Data loading (direct pandas, matching common.py path convention)
# ══════════════════════════════════════════════════════════════════════

def load_symbol_list(data_dir: Path) -> List[str]:
    path = data_dir / "symbol_list.csv"
    if not path.exists():
        raise SystemExit(f"symbol_list.csv not found in {data_dir}")
    syms = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # symbol is typically the first column
        for row in reader:
            if row:
                syms.append(row[0].strip())
    return syms


def load_close_and_cap(data_dir: Path, symbol: str,
                       start: str, end: str) -> Optional[Tuple[pd.Series, pd.Series]]:
    """Load adjusted close and market cap for one symbol. Returns (close, cap) or None."""
    hist = data_dir / "a_shares_history"
    dp_path = hist / f"{symbol}.daily_prices.csv"
    if not dp_path.exists():
        return None
    try:
        dp = pd.read_csv(dp_path, parse_dates=["date"])
    except Exception:
        return None
    if "date" not in dp.columns:
        return None
    dp = dp[(dp["date"] >= start) & (dp["date"] <= end)].sort_values("date")
    if len(dp) < 60:
        return None
    close_col = next((c for c in ["prices.close", "close"] if c in dp.columns), None)
    if close_col is None:
        return None
    dp = dp.set_index("date")
    close = dp[close_col].astype(float)

    # market cap (close * shares); fall back to close if shares unavailable
    cap = close.copy()
    eq_path = hist / f"{symbol}.equity_structures.csv"
    if eq_path.exists():
        try:
            eq = pd.read_csv(eq_path, parse_dates=["date"]).sort_values("date")
            sh_col = next((c for c in ["shares.total", "total_shares",
                                       "shares.circulating", "circulating_shares"]
                           if c in eq.columns), None)
            if sh_col is not None and "date" in eq.columns:
                eq = eq.set_index("date")
                shares = eq[sh_col].astype(float).reindex(close.index, method="ffill")
                cap = close * shares
        except Exception:
            pass
    return close, cap


def load_universe(data_dir: Path, start: str, end: str,
                  max_symbols: int = 4000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    syms = load_symbol_list(data_dir)
    print(f"  symbol_list.csv: {len(syms)} symbols; loading prices {start}..{end}")
    close_d, cap_d = {}, {}
    for i, s in enumerate(syms):
        if len(close_d) >= max_symbols:
            break
        out = load_close_and_cap(data_dir, s, start, end)
        if out is None:
            continue
        close_d[s], cap_d[s] = out
        if (i + 1) % 500 == 0:
            print(f"    scanned {i+1}, loaded {len(close_d)}")
    if not close_d:
        raise SystemExit("No valid price series loaded; check data directory.")
    prices = pd.DataFrame(close_d)
    mcap = pd.DataFrame(cap_d)
    # keep dates where most stocks trade
    prices = prices.ffill()
    valid = prices.notna().mean()
    keep = valid[valid > 0.8].index
    prices, mcap = prices[keep], mcap[keep]
    prices = prices.dropna(how="any")
    mcap = mcap.reindex(prices.index).ffill()
    print(f"  loaded matrix: {prices.shape[1]} stocks x {prices.shape[0]} days")
    return prices, mcap


# ══════════════════════════════════════════════════════════════════════
#  Factor model + constraints
# ══════════════════════════════════════════════════════════════════════

def factor_model_pca(returns: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Sigma = F F^T + diag(D) via PCA. Returns (F (n,k), D (n,))."""
    T, n = returns.shape
    k = min(k, max(1, min(T, n) - 1))
    X = returns - returns.mean(axis=0)
    # SVD of X/sqrt(T): right singular vectors are eigvecs of sample cov
    _, sv, Vt = la.svd(X / math.sqrt(max(T, 1)), full_matrices=False)
    ev = sv[:k] ** 2
    V = Vt[:k].T                       # (n, k)
    F = V * np.sqrt(ev)[None, :]       # (n, k)
    samp_var = X.var(axis=0, ddof=1)
    D = np.maximum(samp_var - (F ** 2).sum(axis=1), 1e-6)
    return F, D


def sector_ids_from_symbols(symbols: List[str]) -> np.ndarray:
    """Rough sector proxy from A-share code prefix (placeholder for SW industry)."""
    def sid(sym: str) -> int:
        c = sym.split(".")[0]
        if c.startswith(("000", "001")): return 0
        if c.startswith(("002", "003")): return 1
        if c.startswith("300"): return 2
        if c.startswith("600"): return 3
        if c.startswith("601"): return 4
        if c.startswith("603"): return 5
        if c.startswith("605"): return 6
        if c.startswith("688"): return 7
        if c.startswith("8"):   return 8
        return 9
    raw = np.array([sid(s) for s in symbols])
    used = sorted(set(raw.tolist()))
    remap = {o: i for i, o in enumerate(used)}
    return np.array([remap[v] for v in raw])


def build_sector_constraints(n: int, sector_ids: np.ndarray, w_bench: np.ndarray,
                             redundant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """E = [budget; sector rows; (redundant budget)], f consistent with w_bench.
    Returns (E, f). Rank deficiency >= 1 because sum(sector rows) == budget row."""
    ns = int(sector_ids.max()) + 1
    S = np.zeros((ns, n))
    for j in range(ns):
        S[j, sector_ids == j] = 1.0
    bs = S @ w_bench
    E_rows = [np.ones(n)]
    f_rows = [1.0]
    for j in range(ns):
        if bs[j] > 1e-12:
            E_rows.append(S[j])
            f_rows.append(float(bs[j]))
    if redundant:
        E_rows.append(np.ones(n))      # duplicate budget -> guaranteed redundancy
        f_rows.append(1.0)
    return np.vstack(E_rows), np.asarray(f_rows)


def _zscore(x: np.ndarray) -> np.ndarray:
    m, s = np.nanmean(x), np.nanstd(x)
    return np.clip((x - m) / (s + 1e-12), -3.0, 3.0)


# ══════════════════════════════════════════════════════════════════════
#  Experiment 1: MINRES vs MINRES+MNR + scaling (real data)
# ══════════════════════════════════════════════════════════════════════

def experiment_scaling(prices: pd.DataFrame, mcap: pd.DataFrame,
                       sizes: List[int], outdir: str, k: int = 10) -> List[Dict]:
    print("\n" + "=" * 72)
    print("  MINRES vs MINRES+MNR  +  matrix-free scaling (real A-shares)")
    print("=" * 72)

    log_p = np.log(prices)
    returns = log_p.diff().dropna()
    n_all = returns.shape[1]
    common = mcap.columns.intersection(returns.columns)
    latest_cap = mcap[common].iloc[-1].dropna()

    rows = []
    print(f"\n  {'n':>5s} {'Erows':>6s} {'rank':>5s} {'red':>4s} "
          f"{'minres_it':>10s} {'relx_r-x_m':>11s} {'||lam_raw||':>12s} "
          f"{'||lam_mnr||':>12s} {'mnr_solve_s':>11s} {'admm_s':>8s} {'status':>10s}")
    print("  " + "-" * 100)

    for n_sub in sizes:
        if n_sub > len(latest_cap):
            continue
        top = latest_cap.nlargest(n_sub).index.tolist()
        rs = returns[top].dropna()
        T, n = rs.shape
        if T < 60 or n < int(n_sub * 0.7):
            print(f"  n={n_sub}: skip (T={T}, n={n})")
            continue

        # factor model on trailing year
        F, D = factor_model_pca(rs.values[-252:], k=k)
        # cap-weighted benchmark
        capv = mcap[top].iloc[-1].fillna(0.0).values.astype(float)
        wb = capv / capv.sum()
        # alpha tilt -> nonzero multiplier
        mom = rs.values[-20:].sum(axis=0)
        vol = rs.values[-60:].std(axis=0)
        alpha = (-0.3 * _zscore(mom) - 0.3 * _zscore(vol))
        alpha = alpha / (np.std(alpha) + 1e-12) * 1e-3

        sids = sector_ids_from_symbols(list(top))
        E, f = build_sector_constraints(n, sids, wb, redundant=True)
        sval = la.svd(E, compute_uv=False)
        rank = int(np.sum(sval > 1e-10 * sval[0]))

        # Factor operator (matrix-free); b tracks benchmark + mild alpha tilt
        A_op = FactorLSOperator(F, D, delta=0.5)
        c = A_op.rhs_for_mu(alpha)            # A^T c = alpha
        b = A_op.matvec(wb) + 0.5 * c

        # ---- equality-constrained inner solve: MINRES vs MINRES+MNR ----
        # tight rtol so the residual is purely null-space; then MNR touches
        # only the multiplier block, not the primal (x_raw == x_mnr)
        t0 = time.time()
        sol = mnr_solve(A_op, b, E, f, rtol=1e-10)
        t_mnr_solve = time.time() - t0
        nx = max(la.norm(sol.x_raw), 1e-12)
        x_diff = float(la.norm(sol.x_raw - sol.x_mnr) / nx)   # RELATIVE
        lam_raw = float(la.norm(sol.lambda_raw))
        lam_mnr = float(la.norm(sol.lambda_mnr))

        # ---- full solve with caps (end-to-end scaling) ----
        cap = 0.02
        t0 = time.time()
        res = admm_mnr(A_op, b, E, f, lower=0.0, upper=cap, tau=2.0, mu_prox=1e-2,
                       outer_tol=1e-3, max_outer=300, completion=True,
                       inner_rtol0=1e-4, inner_power=1.25, inner_floor=1e-8)
        t_admm = res.time
        avg_inner = res.total_kmv / max(len(res.history), 1)

        print(f"  {n:>5d} {E.shape[0]:>6d} {rank:>5d} {E.shape[0]-rank:>4d} "
              f"{sol.iters:>10d} {x_diff:>11.2e} {lam_raw:>12.3e} "
              f"{lam_mnr:>12.3e} {t_mnr_solve:>11.3f} {t_admm:>8.2f} {res.status:>10s}")

        rows.append(dict(n=n, E_rows=E.shape[0], E_rank=rank,
                         redundant=E.shape[0] - rank, minres_iters=sol.iters,
                         x_diff=x_diff, lam_raw=lam_raw, lam_mnr=lam_mnr,
                         mnr_solve_time=t_mnr_solve, admm_time=t_admm,
                         admm_outer=len(res.history), admm_avg_inner=avg_inner,
                         status=res.status))

    # plots
    if rows:
        ns = [r["n"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

        ax[0].plot(ns, [r["minres_iters"] for r in rows], "o-", color="#2196F3", lw=2, ms=8)
        ax[0].set_xlabel("n (assets)"); ax[0].set_ylabel("MINRES iterations (inner solve)")
        ax[0].set_title(f"Inner MINRES count vs n  (k={k})", fontweight="bold")
        ax[0].axhline(np.mean([r["minres_iters"] for r in rows]), color="gray", ls="--", lw=1, label="mean")
        ax[0].legend()

        ax[1].plot(ns, [r["admm_time"] for r in rows], "s-", color="#4CAF50", lw=2, ms=8)
        ax[1].set_xlabel("n"); ax[1].set_ylabel("full solve time (s)")
        ax[1].set_title("ADMM-MNR end-to-end time vs n", fontweight="bold")

        ax[2].plot(ns, [r["lam_raw"] for r in rows], "o-", color="#f44336", lw=2, ms=7, label="||lambda_raw|| (MINRES)")
        ax[2].plot(ns, [r["lam_mnr"] for r in rows], "s-", color="#2196F3", lw=2, ms=7, label="||lambda_mnr|| (MINRES+MNR)")
        ax[2].set_xlabel("n"); ax[2].set_ylabel("multiplier norm")
        ax[2].set_title("Multiplier: MINRES vs MINRES+MNR", fontweight="bold")
        ax[2].legend(fontsize=8)

        fig.tight_layout()
        p = os.path.join(outdir, "minres_vs_mnr_scaling.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\n  -> {p}")
    return rows


# ══════════════════════════════════════════════════════════════════════
#  Experiment 2: floating-point inconsistency (where MNR matters)
# ══════════════════════════════════════════════════════════════════════

def experiment_inconsistency(prices: pd.DataFrame, mcap: pd.DataFrame,
                             outdir: str, n_sub: int = 300, k: int = 10) -> List[Dict]:
    print("\n" + "=" * 72)
    print("  Floating-point inconsistency in sector targets (MNR robustness)")
    print("=" * 72)

    log_p = np.log(prices); returns = log_p.diff().dropna()
    common = mcap.columns.intersection(returns.columns)
    latest_cap = mcap[common].iloc[-1].dropna()
    n_sub = min(n_sub, len(latest_cap))
    top = latest_cap.nlargest(n_sub).index.tolist()
    rs = returns[top].dropna()
    n = rs.shape[1]

    F, D = factor_model_pca(rs.values[-252:], k=k)
    A_op = FactorLSOperator(F, D, delta=0.5)
    capv = mcap[top].iloc[-1].fillna(0.0).values.astype(float)
    wb = capv / capv.sum()
    alpha = _zscore(rs.values[-20:].sum(axis=0)) * 1e-3
    c = A_op.rhs_for_mu(alpha)
    b = A_op.matvec(wb) + 2.0 * c
    sids = sector_ids_from_symbols(list(top))

    print(f"\n  n={n}, perturbing one sector target by delta (breaks sum==budget)")
    print(f"  {'delta':>8s} {'||x_raw-x_mnr||':>16s} {'feasible_x':>11s} "
          f"{'||lam_raw||':>12s} {'||lam_mnr||':>12s}")
    print("  " + "-" * 64)

    rows = []
    for delta in [0.0, 1e-6, 1e-4, 1e-2]:
        E, f = build_sector_constraints(n, sids, wb, redundant=False)
        f = f.copy(); f[1] += delta   # perturb first sector target
        sol = mnr_solve(A_op, b, E, f, rtol=1e-10)
        fR = row_project_rhs(E, f)
        feas = float(la.norm(E @ sol.x_mnr - fR))
        xdiff = float(la.norm(sol.x_raw - sol.x_mnr))
        lr, lm = float(la.norm(sol.lambda_raw)), float(la.norm(sol.lambda_mnr))
        print(f"  {delta:>8.0e} {xdiff:>16.2e} {feas:>11.2e} {lr:>12.3e} {lm:>12.3e}")
        rows.append(dict(delta=delta, x_diff=xdiff, feasible=feas,
                         lam_raw=lr, lam_mnr=lm))

    if rows:
        deltas = [max(r["delta"], 1e-8) for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.loglog(deltas, [r["lam_raw"] for r in rows], "o-", color="#f44336",
                  lw=2, ms=8, label="||lambda_raw|| (MINRES)")
        ax.loglog(deltas, [r["lam_mnr"] for r in rows], "s-", color="#2196F3",
                  lw=2, ms=8, label="||lambda_mnr|| (MINRES+MNR)")
        ax.set_xlabel("sector-target inconsistency delta")
        ax.set_ylabel("multiplier norm")
        ax.set_title("MNR keeps the multiplier bounded under inconsistency", fontweight="bold")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(outdir, "minres_vs_mnr_inconsistency.png")
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"\n  -> {p}")
    return rows


# ══════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description="MINRES vs MINRES+MNR on A-share factor model")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="crawler data dir (symbol_list.csv + a_shares_history/)")
    ap.add_argument("--outdir", type=str, default="results_minres_vs_mnr")
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 500, 1000, 2000, 3000])
    ap.add_argument("--k", type=int, default=10, help="number of factors")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2024-12-31")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    prices, mcap = load_universe(args.data_dir, args.start, args.end,
                                 max_symbols=max(args.sizes) + 200)

    scaling_rows = experiment_scaling(prices, mcap, args.sizes, args.outdir, k=args.k)
    incons_rows = experiment_inconsistency(prices, mcap, args.outdir,
                                           n_sub=min(args.sizes), k=args.k)

    # write CSV summaries
    def _write(path, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow(r)
    _write(os.path.join(args.outdir, "scaling_summary.csv"), scaling_rows)
    _write(os.path.join(args.outdir, "inconsistency_summary.csv"), incons_rows)
    print(f"\nDone. Figures and CSVs in {args.outdir}/")


if __name__ == "__main__":
    main()
