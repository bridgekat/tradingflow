from dataclasses import dataclass, field

import numpy as np
import scipy as sp


@dataclass(slots=True)
class MinimumVarianceState:
    #: GMV weights of the predicted covariance, zero outside the eligible
    #: block. `None` until the first prediction.
    weights: np.ndarray | None = None
    sum_r: float = 0.0
    sum_r_sq: float = 0.0
    count: int = 0
    out: float = field(default=float("nan"))

    def set_predict(self, sigma_log: np.ndarray) -> None:
        """Caches the GMV weights of a log-return covariance matrix.

        The prediction is a log-return covariance, converted to linear-return
        units before solving GMV (the zero-mean specialisation of the
        lognormal moment map), so the weights minimise variance in the same
        units the realized variance is later reported in.

        Stocks with a non-finite covariance diagonal are excluded and keep a
        zero weight.
        """
        mask = np.isfinite(np.diag(sigma_log))
        weights = np.zeros(sigma_log.shape[0])
        if mask.any():
            block = sigma_log[np.ix_(mask, mask)]
            factor = 1.0 + np.expm1(0.5 * np.diag(block))
            weights[mask] = gmv_weights(np.outer(factor, factor) * np.expm1(block))
        self.weights = weights


def gmv_weights(sigma: np.ndarray) -> np.ndarray:
    r"""Global minimum variance weights of a covariance matrix.

    The closed form of `minimize w' Σ w subject to 1' w = 1`, namely
    \(w = \Sigma^+ \mathbf{1} / (\mathbf{1}^T \Sigma^+ \mathbf{1})\). No
    non-negativity constraint, so shorts are allowed.

    Uses the pseudo-inverse rather than a direct solve, which handles
    rank-deficient covariance (sample covariance with `N > T`, say) by taking
    the minimum-norm solution. A zero denominator yields non-finite weights.
    """
    ones = np.ones(sigma.shape[0])
    w = sp.linalg.pinv(sigma) @ ones
    return w / (ones @ w)


class MinimumVariance:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, float]
    type Context = int
    type State = MinimumVarianceState

    def __init__(self):
        pass

    def init(self, _: Inputs) -> State:
        return MinimumVarianceState()

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return (False, state.out)

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        predict_signal, predict, target_signal, target = inputs

        if target_signal and state.weights is not None:
            # Fold one portfolio return per sampling period. The target is a
            # log return, mapped elementwise to the linear units the GMV
            # objective minimises; non-finite entries contribute zero.
            r = np.expm1(np.where(np.isfinite(target), target, 0.0))
            r_p = float(state.weights @ r)
            state.sum_r += r_p
            state.sum_r_sq += r_p * r_p
            state.count += 1

        if predict_signal:
            # A new prediction closes the open evaluation period and opens the
            # next. The first one has no period to close.
            emit = state.weights is not None
            if emit:
                mean = state.sum_r / max(state.count, 1)
                state.out = state.sum_r_sq / max(state.count, 1) - mean * mean
            state.set_predict(predict)
            state.sum_r = 0.0
            state.sum_r_sq = 0.0
            state.count = 0
            return (emit, state.out)
        else:
            return (False, state.out)


__op__ = MinimumVariance()
