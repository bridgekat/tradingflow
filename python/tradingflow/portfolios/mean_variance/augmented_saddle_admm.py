"""
augmented_saddle_admm.py
========================

带等式约束的 mean-variance 组合优化求解核心. 单一信息源, 被 multi_account.py
和 sector_constrained.py 共享.

求解问题:

    min_x   (delta/1) * || A x ||^2 - mu^T x
    s.t.    C x = d                           [等式约束]
            x >= 0  (可选)                    [long-only]

外层用 ADMM 拆分非负约束, 内层把等式约束写成 augmented saddle form 后用
MINRES + Minimum-Norm Refinement (MNR) 求解.

跟传统 direct LU on Schur form 的关键差别: **C 矩阵不要求行满秩**.
当 C 有冗余行 (e.g., 多账户场景下 firm-level 总预算跟单账户预算之和重合,
或 sector 约束之和跟总预算重合), 传统 LU 对应的 KKT 矩阵奇异返回 NaN;
我们的 augmented saddle + MNR 框架对此完全透明.

References
----------
- Liu, Cartis, Hauser. "Augmented Saddle ADMM for Singular Symmetric Least
  Squares: Convergence Rates, Minimum-Norm Refinement, and Active-Set
  Identification" (SISC submission).
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.linalg as spla

# =====================================================================
# 1. Paper 的 MINRES 实现 (recurrence-based ||K r||/||K rhs|| stopping)
# =====================================================================


def _symmetric_givens(a: float, b: float) -> Tuple[float, float, float]:
    a = float(a)
    b = float(b)
    if b == 0.0:
        c = 1.0 if a == 0.0 else float(np.sign(a))
        return c, 0.0, abs(a)
    if a == 0.0:
        return 0.0, float(np.sign(b)), abs(b)
    if abs(b) > abs(a):
        t = a / b
        s = np.sign(b) / np.sqrt(1.0 + t * t)
        c = s * t
        r = b / s
    else:
        t = b / a
        c = np.sign(a) / np.sqrt(1.0 + t * t)
        s = c * t
        r = a / c
    return c, s, r


@dataclass
class _MinresResult:
    iterate: np.ndarray
    residual: np.ndarray
    iterations: int
    relative_Ks: float


def _minres_paper(
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    rtol: float = 1e-6,
    maxit: Optional[int] = None,
    reorthogonalize: bool = True,
    zero_tol: float = 1e-12,
) -> _MinresResult:
    """
    Paper MINRES with recurrence-based normal-residual stopping rule
    ||K r_{k-1}|| / ||K * rhs|| <= rtol.

    Unlike scipy's MINRES which uses ||r||/||b||, this criterion stops
    when the *range* component of residual is small, deliberately leaving
    any *null space* component for the rank-one MNR step to handle.
    For non-singular K (consistent system), MINRES drives the full residual
    to machine precision and MNR is a no-op.
    """
    n = rhs.shape[0]
    if maxit is None:
        maxit = 4 * n

    r2 = rhs.copy()
    r3 = r2.copy()
    beta1 = float(np.linalg.norm(r2))
    if beta1 == 0.0:
        return _MinresResult(np.zeros(n), np.zeros(n), 0, 0.0)

    flag = -2
    beta = 0.0
    c, s = -1.0, 0.0
    delta_next, epsilon_next = 0.0, 0.0
    phi = beta1
    residual = rhs.copy()
    r1 = np.zeros(n)
    v_next = r3 / beta1
    V = v_next.reshape(-1, 1)
    K_rhs_norm = float(np.linalg.norm(matvec(rhs)))

    x = np.zeros(n)
    w, w_prev = np.zeros(n), np.zeros(n)

    iterations = 0
    previous_x = x.copy()
    previous_residual = residual.copy()
    previous_phi = phi
    relative_Ks = np.inf
    beta_next = beta1

    while flag == -2:
        beta_previous = beta
        beta = beta_next
        v = v_next.copy()
        r3 = matvec(v)
        iterations += 1
        alpha = float(np.dot(r3, v))

        if iterations > 1:
            r3 = r3 - r1 * beta / beta_previous
        r3 = r3 - r2 * alpha / beta

        if reorthogonalize:
            r3 = r3 - V @ (V.T @ r3)

        r1 = r2.copy()
        r2 = r3.copy()
        beta_next = float(np.linalg.norm(r3))

        if iterations == 1 and beta_next < zero_tol:
            if alpha == 0.0:
                break
            x = rhs / alpha
            previous_x = x.copy()
            previous_residual = np.zeros_like(rhs)
            previous_phi = 0.0
            relative_Ks = 0.0
            break

        if beta_next > zero_tol:
            v_next = r3 / beta_next
        else:
            v_next = np.zeros_like(rhs)

        if reorthogonalize and beta_next > zero_tol:
            V = np.hstack([V, v_next.reshape(-1, 1)])

        delta_bar = delta_next
        delta = c * delta_bar + s * alpha
        epsilon = epsilon_next
        gamma_bar = s * delta_bar - c * alpha
        epsilon_next = s * beta_next
        delta_next = -c * beta_next

        previous_phi = phi
        previous_residual = residual.copy()
        previous_x = x.copy()

        c, s, gamma = _symmetric_givens(gamma_bar, beta_next)

        if gamma > zero_tol:
            tau = c * phi
            phi = s * phi
            w_older = w_prev.copy()
            w_prev = w.copy()
            w = (v - epsilon * w_older - delta * w_prev) / gamma
            x = x + tau * w
            residual = (s**2) * previous_residual - phi * c * v_next
        else:
            c, s = 0.0, 1.0
            phi = previous_phi
            residual = previous_residual.copy()
            x = previous_x.copy()
            flag = 2
            maxit += 1

        Ks_previous = previous_phi * np.sqrt(gamma_bar**2 + delta_next**2)
        if K_rhs_norm > 0.0:
            relative_Ks = Ks_previous / K_rhs_norm
        else:
            relative_Ks = 0.0

        if relative_Ks <= rtol:
            flag = 5
        if iterations >= maxit:
            flag = 4

    return _MinresResult(
        iterate=previous_x,
        residual=previous_residual,
        iterations=iterations,
        relative_Ks=float(relative_Ks),
    )


def _apply_mnr_step(iterate: np.ndarray, residual: np.ndarray, threshold: float) -> Tuple[np.ndarray, bool]:
    """
    一秩 MNR 修正: w <- w - (<r, w>/||r||^2) * r.

    数学意义: 当 K 有 null space 且 residual 落在 null(K) 里, 此修正给出
    minimum-norm 意义下的正确解 (paper Prop 1). 对 non-singular K 是恒等.

    threshold: 数值安全网, 当 ||residual|| <= threshold 时跳过修正
    (防止一致系统已收敛到机器精度时的 0/0 数值爆炸). Paper 默认 1e-10 * ||rhs||.

    Note: 在 tradingflow 的典型场景 (一致系统 + 冗余约束) 下, 此函数通常返回
    iterate 原值 (skip 路径), MNR 作为安全网存在但不实际修改解. 真正发力
    的场景是 inconsistent 系统 (paper §11), tradingflow 当前工作流中不会出现.
    """
    res_norm = float(np.linalg.norm(residual))
    if res_norm <= threshold:
        return iterate, False
    res_norm_sq = res_norm * res_norm
    coef = float(np.dot(residual, iterate)) / res_norm_sq
    refined = iterate - coef * residual
    return refined, True


# =====================================================================
# 2. ADMM 外层 (通用, 由调用方提供 K_times 和 rhs_builder)
# =====================================================================


@dataclass
class ADMMResult:
    x: np.ndarray
    lam: np.ndarray
    outer_iters: int
    inner_iters_total: int
    mnr_fires: int
    primal_residual: float
    dual_residual: float


def solve_admm(
    K_times: Callable[[np.ndarray], np.ndarray],
    rhs_builder: Callable[[np.ndarray], np.ndarray],
    n_residual_block: int,
    n_x_block: int,
    n_lambda_block: int,
    *,
    rho: float = 1.0,
    project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    inner_rtol: float = 1e-6,
    mnr_threshold_rel: float = 1e-10,
    outer_tol: float = 1e-6,
    max_outer: int = 2000,
    alpha_over: float = 1.7,
) -> ADMMResult:
    """
    通用 augmented saddle ADMM 外层.

    Parameters
    ----------
    K_times : matvec 函数, 作用于 augmented saddle 全空间 (r; x; lam)
    rhs_builder : 给 ADMM 状态 v, 返回 rhs 向量
    n_residual_block, n_x_block, n_lambda_block : iterate 的三块大小
        约定 iterate = (r; x; lam),
        总维数 = n_residual_block + n_x_block + n_lambda_block
    rho : ADMM 罚参数 (默认 1.0)
    project_fn : 自定义 z-update 投影. 给定 `x_hat + u`, 返回投影后的 z.
        若为 None (默认), z-update 不做投影 (等式约束-only 求解).
        long-only 的典型用法: project_fn=lambda z: np.maximum(z, 0.0).
        x_aug 内部混合了 "primary + 各种 slack" 等需要分块差异化投影
        的场景, 由调用方写自定义 callable.
    inner_rtol : MINRES 停机相对阈值 ||K r||/||K rhs|| <= rtol
    mnr_threshold_rel : MNR 跳过阈值 (相对 ||rhs||), paper 默认 1e-10
    outer_tol : ADMM primal/dual 残差停机阈值
    max_outer : 外层最大迭代
    alpha_over : ADMM over-relaxation (默认 1.7)
    """
    nx = n_x_block
    nr = n_residual_block

    z = np.zeros(nx)
    u = np.zeros(nx)
    total_inner = 0
    mnr_fires = 0
    primal_res, dual_res = np.inf, np.inf
    last_lam = np.zeros(n_lambda_block)

    for k_outer in range(max_outer):
        v_admm = z - u
        rhs = rhs_builder(v_admm)
        rhs_norm = float(np.linalg.norm(rhs))

        result = _minres_paper(K_times, rhs, rtol=inner_rtol)
        total_inner += result.iterations

        iterate = result.iterate
        threshold = mnr_threshold_rel * rhs_norm
        iterate, fired = _apply_mnr_step(iterate, result.residual, threshold)
        if fired:
            mnr_fires += 1

        x_new = iterate[nr : nr + nx]
        last_lam = iterate[nr + nx :].copy()

        x_hat = alpha_over * x_new + (1.0 - alpha_over) * z
        if project_fn is not None:
            z_new = project_fn(x_hat + u)
        else:
            z_new = x_hat + u
        u_new = u + x_hat - z_new

        primal_res = float(np.linalg.norm(x_new - z_new))
        dual_res = rho * float(np.linalg.norm(z_new - z))

        z = z_new
        u = u_new
        if primal_res < outer_tol and dual_res < outer_tol:
            break

    return ADMMResult(
        x=z.copy(),
        lam=last_lam,
        outer_iters=k_outer + 1,
        inner_iters_total=total_inner,
        mnr_fires=mnr_fires,
        primal_residual=primal_res,
        dual_residual=dual_res,
    )


# =====================================================================
# 3. 场景 setup 工具: 把高层问题转成 K_times + rhs_builder
# =====================================================================


def setup_multi_account(
    sigma: np.ndarray,
    mus: List[np.ndarray],  # K 个账户的 mu 向量
    deltas: List[float],  # K 个账户的 delta 风险厌恶
    account_budgets: List[float],  # K 个账户预算, sum = 1
    include_firm_budget: bool = True,
    rho: float = 1.0,
) -> Tuple[Callable, Callable, dict]:
    """
    构造 K 账户联合优化的 augmented saddle 表示.

    Returns
    -------
    K_times, rhs_builder : 给 solve_admm 用
    info : dict, 含 n_residual_block, n_x_block, n_lambda_block
        以及辅助信息 (n_per_account, K, etc.)
    """
    K = len(mus)
    n = sigma.shape[0]
    assert len(deltas) == K and len(account_budgets) == K
    for mu in mus:
        assert mu.shape == (n,)
    if include_firm_budget:
        assert abs(sum(account_budgets) - 1.0) < 1e-6, "账户预算之和必须 = 1"

    # Cholesky of sigma
    L = spla.cholesky(sigma + 1e-14 * np.eye(n), lower=True)
    sqrt_2d = np.array([np.sqrt(2.0 * d) for d in deltas])

    # b 向量: A^T b = mu_k 对每个账户, 然后拼起来
    b_blocks = [spla.solve_triangular(L, mus[k], lower=True) / sqrt_2d[k] for k in range(K)]
    b_inner = np.concatenate(b_blocks)

    def A_times(x_all):
        # x_all = (x_1; ... ; x_K), each (n,)
        out = np.empty(K * n)
        for k in range(K):
            out[k * n : (k + 1) * n] = sqrt_2d[k] * (L.T @ x_all[k * n : (k + 1) * n])
        return out

    def AT_times(r_all):
        out = np.empty(K * n)
        for k in range(K):
            out[k * n : (k + 1) * n] = sqrt_2d[k] * (L @ r_all[k * n : (k + 1) * n])
        return out

    # 约束矩阵 C 的 matvec:
    #   K 行: 单账户预算 1^T x_k = budgets[k]
    #   (可选 1 行: firm budget 1^T sum_k x_k = 1, 跟前 K 行线性相关)
    m_constraints = K + 1 if include_firm_budget else K
    budgets_array = np.array(account_budgets)

    def C_times(x_all):
        out = np.empty(m_constraints)
        for k in range(K):
            out[k] = x_all[k * n : (k + 1) * n].sum()
        if include_firm_budget:
            out[K] = x_all.sum()
        return out

    def CT_times(lam):
        out = np.empty(K * n)
        firm_part = lam[K] if include_firm_budget else 0.0
        for k in range(K):
            out[k * n : (k + 1) * n] = lam[k] + firm_part
        return out

    if include_firm_budget:
        d_constraint = np.concatenate([budgets_array, [1.0]])
    else:
        d_constraint = budgets_array.copy()

    nx = K * n
    nr = K * n

    def K_times(w):
        r = w[:nr]
        x = w[nr : nr + nx]
        lam = w[nr + nx :]
        out = np.empty(nr + nx + m_constraints)
        out[:nr] = r + A_times(x)
        out[nr : nr + nx] = AT_times(r) - rho * x - CT_times(lam)
        out[nr + nx :] = -C_times(x)
        return out

    def rhs_builder(v):
        out = np.empty(nr + nx + m_constraints)
        out[:nr] = b_inner
        out[nr : nr + nx] = -rho * v
        out[nr + nx :] = -d_constraint
        return out

    return (
        K_times,
        rhs_builder,
        {
            "n_residual_block": nr,
            "n_x_block": nx,
            "n_lambda_block": m_constraints,
            "n_per_account": n,
            "K": K,
        },
    )


def setup_sector_constrained(
    sigma: np.ndarray,
    mu: np.ndarray,
    delta: float,
    sector_membership: List[List[int]],  # K 个行业各自的资产 index 列表
    sector_targets: List[float],  # K 个行业目标权重, sum = 1
    include_total_budget: bool = True,
    rho: float = 1.0,
) -> Tuple[Callable, Callable, dict]:
    """
    构造 sector-constrained 单账户优化的 augmented saddle 表示.
    """
    n = sigma.shape[0]
    K = len(sector_membership)
    assert len(sector_targets) == K
    if include_total_budget:
        assert abs(sum(sector_targets) - 1.0) < 1e-6, "行业目标权重之和必须 = 1"

    # 验证 sector 覆盖
    all_assets = set()
    for idxs in sector_membership:
        for i in idxs:
            assert 0 <= i < n
            assert i not in all_assets, f"资产 {i} 属于多个行业"
            all_assets.add(i)
    assert len(all_assets) == n, "行业未穷尽 universe"

    L = spla.cholesky(sigma + 1e-14 * np.eye(n), lower=True)
    sqrt_2d = np.sqrt(2.0 * delta)
    b_inner = spla.solve_triangular(L, mu, lower=True) / sqrt_2d

    # Sector indicator matrix
    S = np.zeros((K, n))
    for k, idxs in enumerate(sector_membership):
        S[k, idxs] = 1.0

    m_constraints = K + 1 if include_total_budget else K

    def A_times(x):
        return sqrt_2d * (L.T @ x)

    def AT_times(r):
        return sqrt_2d * (L @ r)

    def C_times(x):
        out = np.empty(m_constraints)
        out[:K] = S @ x
        if include_total_budget:
            out[K] = x.sum()
        return out

    def CT_times(lam):
        out = S.T @ lam[:K]
        if include_total_budget:
            out = out + lam[K]
        return out

    if include_total_budget:
        d_constraint = np.concatenate([np.array(sector_targets), [1.0]])
    else:
        d_constraint = np.array(sector_targets)

    nx = n
    nr = n

    def K_times(w):
        r = w[:nr]
        x = w[nr : nr + nx]
        lam = w[nr + nx :]
        out = np.empty(nr + nx + m_constraints)
        out[:nr] = r + A_times(x)
        out[nr : nr + nx] = AT_times(r) - rho * x - CT_times(lam)
        out[nr + nx :] = -C_times(x)
        return out

    def rhs_builder(v):
        out = np.empty(nr + nx + m_constraints)
        out[:nr] = b_inner
        out[nr : nr + nx] = -rho * v
        out[nr + nx :] = -d_constraint
        return out

    return (
        K_times,
        rhs_builder,
        {
            "n_residual_block": nr,
            "n_x_block": nx,
            "n_lambda_block": m_constraints,
            "sector_indicators": S,
            "K_sectors": K,
        },
    )


# =====================================================================
# 4. 自适应配比 / sector rotation 辅助函数
# =====================================================================


def sigmoid_split(
    signal: float,
    *,
    baseline: float = 0.0,
    scale: float = 1.0,
    min_w: float = 0.2,
    max_w: float = 0.8,
) -> float:
    """把一个标量信号映射到 [min_w, max_w], 中性时返回 (min_w + max_w)/2."""
    normalized = (signal - baseline) / scale
    if normalized >= 0:
        sig = 1.0 / (1.0 + np.exp(-normalized))
    else:
        e = np.exp(normalized)
        sig = e / (1.0 + e)
    return min_w + (max_w - min_w) * float(sig)


def adaptive_two_account_split(
    signal: float,
    baseline: float = 0.0,
    scale: float = 1.0,
    min_aggressive: float = 0.2,
    max_aggressive: float = 0.8,
) -> Tuple[float, float]:
    """两账户配比映射: 返回 (w_conservative, w_aggressive)."""
    w_B = sigmoid_split(signal, baseline=baseline, scale=scale, min_w=min_aggressive, max_w=max_aggressive)
    return 1.0 - w_B, w_B


def adaptive_sector_weights(
    signals: np.ndarray,
    baseline: np.ndarray,
    signal_strength: float = 1.0,
    min_weight: float = 0.05,
    max_weight: float = 0.50,
) -> np.ndarray:
    """Sector rotation: 把行业信号映射到调整后的目标权重, 满足 sum=1 和上下限."""
    centered = signals - signals.mean()
    std = signals.std() + 1e-12
    log_w = np.log(baseline + 1e-12) + signal_strength * (centered / std)
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    w = np.clip(w, min_weight, max_weight)
    w = w / w.sum()
    return w
