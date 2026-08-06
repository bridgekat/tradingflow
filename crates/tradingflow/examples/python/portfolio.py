from dataclasses import dataclass
import sys
import numpy as np
import cvxpy as cp


@dataclass(slots=True)
class MarkowitzSolver:
    """A Markowitz mean-variance portfolio optimizer using CVXPY DPP.
    It solves the convex optimization problem:

        max_w { μᵀ w - δ wᵀ Σ w }

    where `Σ = X F Xᵀ + D`, `F` is symmetric positive semi-definite, `X` is
    (narrow) rectangular, and `D` is diagonal with non-negative entries.
    `δ` is the risk aversion parameter.
    """

    n: int
    k: int
    w: cp.Variable
    max: cp.Parameter
    bench: cp.Parameter
    mu: cp.Parameter
    rank: cp.Parameter  # any matrix `Z` such that `X F Xᵀ = Z Zᵀ`
    diag: cp.Parameter  # square root of diagonal of `D`
    problem: cp.Problem

    def __init__(
        self,
        n: int,
        k: int,
        benchmark_relative: bool,
        risk_aversion: float,
        long_only: bool,
        full_position: bool,
    ) -> None:
        """Initializes the optimizer for `n` assets and rank-`k` risk factor model."""

        self.n = n
        self.k = k
        self.w = cp.Variable(n)
        self.max = cp.Parameter(n, nonneg=True)
        self.bench = cp.Parameter(n)
        self.mu = cp.Parameter(n)
        self.rank = cp.Parameter((n, k))
        self.diag = cp.Parameter(n)

        constraints: list[cp.Constraint] = []
        constraints.append(cp.abs(self.w) <= self.max)  # max position size
        constraints.append(cp.norm1(self.w) <= 1.0)  # no leverage
        if long_only:
            constraints.append(self.w >= 0.0)  # no short-selling
        if full_position:
            constraints.append(self.w.sum() == 1.0)  # fully invested

        if benchmark_relative:
            active_w = self.w - self.bench
            active_returns = self.mu.T @ active_w
            tracking_error = cp.sum_squares(self.rank.T @ active_w) + cp.sum_squares(
                cp.multiply(self.diag, active_w)
            )
            objective = cp.Maximize(active_returns - risk_aversion * tracking_error)
        else:
            returns = self.mu.T @ self.w
            variance = cp.sum_squares(self.rank.T @ self.w) + cp.sum_squares(
                cp.multiply(self.diag, self.w)
            )
            objective = cp.Maximize(returns - risk_aversion * variance)

        self.problem = cp.Problem(objective, constraints)

    def solve(
        self,
        max: np.ndarray,
        bench: np.ndarray,
        mu: np.ndarray,
        exposures: np.ndarray,  # X
        covariance: np.ndarray,  # F
        specific: np.ndarray,  # diagonal of D
    ) -> np.ndarray | None:
        """Solves the optimization problem and returns the optimal weights."""

        assert max.shape == (self.n,) and np.isfinite(max).all()
        assert bench.shape == (self.n,) and np.isfinite(bench).all()
        assert mu.shape == (self.n,) and np.isfinite(mu).all()
        assert exposures.shape == (self.n, self.k) and np.isfinite(exposures).all()
        assert covariance.shape == (self.k, self.k) and np.isfinite(covariance).all()
        assert specific.shape == (self.n,) and np.isfinite(specific).all()

        lam, s = np.linalg.eigh(covariance)  # F = S Λ S⁻¹ = S Λ Sᵀ
        rank = exposures @ s @ np.diag(np.sqrt(np.maximum(lam, 0.0)))
        diag = np.sqrt(np.maximum(specific, 0.0))

        self.max.value = max
        self.bench.value = bench
        self.mu.value = mu
        self.rank.value = rank
        self.diag.value = diag

        try:
            self.problem.solve(solver=cp.SCS, warm_start=True)
            if self.w.value is not None:
                return self.w.value
            else:
                print("portfolio: no solution", file=sys.stderr)
                return None

        except cp.SolverError:
            print("portfolio: solver failed", file=sys.stderr)
            return None


@dataclass(slots=True)
class PortfolioState:
    solver: MarkowitzSolver
    out_weights: np.ndarray


class Portfolio:
    type Inputs = tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]
    type Outputs = np.ndarray
    type Context = int
    type State = PortfolioState

    def __init__(
        self,
        benchmark_relative: bool,
        risk_aversion: float,
        long_only: bool,
        full_position: bool,
    ) -> None:
        assert risk_aversion > 0.0, "portfolio: risk_aversion must be positive"

        self.benchmark_relative = benchmark_relative
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.full_position = full_position

    def init(self, inputs: Inputs) -> State:
        rebalance_signal, index_weights, mu, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert index_weights.shape == (n,)
        assert mu.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        return PortfolioState(
            solver=MarkowitzSolver(
                n=n,
                k=k,
                benchmark_relative=self.benchmark_relative,
                risk_aversion=self.risk_aversion,
                long_only=self.long_only,
                full_position=self.full_position,
            ),
            out_weights=np.zeros((n,)),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_weights

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        rebalance_signal, index_weights, mu, exposures, covariance, specific = inputs
        n, k = exposures.shape
        assert index_weights.shape == (n,)
        assert mu.shape == (n,)
        assert covariance.shape == (k, k)
        assert specific.shape == (n,)

        if rebalance_signal:
            valid = (
                (index_weights > 0.0)
                & np.isfinite(mu)
                & np.isfinite(exposures).all(axis=1)
                & np.isfinite(specific)
            )
            weights = state.solver.solve(
                max=np.where(valid, 1.0, 0.0),
                bench=np.where(valid, index_weights, 0.0),
                mu=np.where(valid, mu, 0.0),
                exposures=np.where(valid[:, None], exposures, 0.0),
                covariance=covariance,
                specific=np.where(valid, specific, 0.0),
            )
            if weights is not None:
                state.out_weights = weights

        return state.out_weights


def build(**kwargs) -> Portfolio:
    return Portfolio(**kwargs)
