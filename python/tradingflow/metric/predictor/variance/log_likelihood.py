from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class LogLikelihoodState:
    #: Pseudo-inverse of the predicted covariance, zero outside the eligible
    #: block. `None` until the first prediction.
    precision: np.ndarray | None = None
    log_det: float = 0.0
    sum_quad: float = 0.0
    count: int = 0
    out: float = field(default=float("nan"))

    def set_predict(self, sigma: np.ndarray) -> None:
        """Caches the precision matrix and log-determinant of `sigma`.

        Stocks with a non-finite covariance diagonal are excluded; their rows
        and columns of the precision matrix stay zero, so their targets
        contribute nothing to the quadratic form.
        """
        mask = np.isfinite(np.diag(sigma))
        precision = np.zeros_like(sigma)
        log_det = 0.0
        if mask.any():
            block = np.ix_(mask, mask)
            log_det, precision[block] = log_pdet_and_pinv(sigma[block])
        self.precision = precision
        self.log_det = log_det


def log_pdet_and_pinv(sigma: np.ndarray) -> tuple[float, np.ndarray]:
    r"""Log-pseudo-determinant and Moore-Penrose pseudo-inverse via one SVD.

    For a real symmetric matrix the singular values equal the absolute
    eigenvalues (and the eigenvalues themselves when PSD), so one SVD of the
    symmetrized `sigma` yields both: the log-pseudo-determinant is
    \(\sum_{s_i > \text{cutoff}} \log s_i\), and the pseudo-inverse is
    \(V \operatorname{diag}(1 / s) U^T\) on the retained subspace, zero
    elsewhere.

    The cutoff \(\max(M, N) \cdot \varepsilon \cdot s_\max\) matches scipy's
    default `rcond`, so numerical-artifact negative eigenvalues (from
    pairwise-deletion sample covariance, say) fall below it and are discarded
    — correctly restricting the likelihood to the PSD subspace.
    """
    sym = 0.5 * (sigma + sigma.T)
    u, s, vt = np.linalg.svd(sym)
    cutoff = max(sym.shape) * np.finfo(sym.dtype).eps * (float(s[0]) if s.size else 0.0)
    mask = s > cutoff
    log_pdet = float(np.log(s[mask]).sum()) if mask.any() else 0.0
    inv_s = np.where(mask, 1.0 / np.where(mask, s, 1.0), 0.0)
    return log_pdet, (vt.T * inv_s) @ u.T


class LogLikelihood:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, float]
    type Context = int
    type State = LogLikelihoodState

    def __init__(self):
        pass

    def init(self, _: Inputs) -> State:
        return LogLikelihoodState()

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return (False, state.out)

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        predict_signal, predict, target_signal, target = inputs

        if target_signal and state.precision is not None:
            # Fold one Mahalanobis quadratic per sampling period. Non-finite
            # targets contribute zero.
            r = np.where(np.isfinite(target), target, 0.0)
            state.sum_quad += float(r @ state.precision @ r)
            state.count += 1

        if predict_signal:
            # A new prediction closes the open evaluation period and opens the
            # next. The first one has no period to close.
            emit = state.precision is not None
            if emit:
                state.out = state.log_det + state.sum_quad / max(state.count, 1)
            state.set_predict(predict)
            state.sum_quad = 0.0
            state.count = 0
            return (emit, state.out)
        else:
            return (False, state.out)


__op__ = LogLikelihood()
