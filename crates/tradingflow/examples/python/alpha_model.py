from dataclasses import dataclass
from collections import deque
import sys
import numpy as np


@dataclass(slots=True)
class RidgeRegressor:
    """An incremental ridge regression model for mean (alpha) prediction.
    It solves the L2-regularized least-squares problem:

        min_β { (1 / m) ‖X β - y‖² + δ ‖β‖² }

    where `m` is the number of rows in `X` and `y`, each row being one
    pool-standardized sample. `δ` is the regularization parameter.
    """

    count: int
    sum_x: np.ndarray
    sum_xx: np.ndarray
    sum_y: float
    sum_yy: float
    sum_xy: np.ndarray

    mean_x: np.ndarray
    mean_y: float
    std_x: np.ndarray
    std_y: float
    beta: np.ndarray

    def __init__(self, k: int) -> None:
        """Initializes the model for `k` features."""

        self.count = 0
        self.sum_x = np.zeros(k)
        self.sum_xx = np.zeros((k, k))
        self.sum_y = 0.0
        self.sum_yy = 0.0
        self.sum_xy = np.zeros(k)

        self.mean_x = np.zeros(k)
        self.mean_y = 0.0
        self.std_x = np.ones(k)
        self.std_y = 1.0
        self.beta = np.zeros(k)

    def add_samples(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Adds a batch of samples `(xs, ys)` to the accumulator."""

        m = xs.shape[0]
        assert xs.shape == (m, self.sum_x.shape[0]) and np.isfinite(xs).all()
        assert ys.shape == (m,) and np.isfinite(ys).all()

        self.count += m
        self.sum_x += np.sum(xs, axis=0)
        self.sum_xx += xs.T @ xs
        self.sum_y += np.sum(ys)
        self.sum_yy += np.sum(ys * ys)
        self.sum_xy += xs.T @ ys

    def fit(self, delta: float) -> None:
        """Solves the normal equation `(Xᵀ X + m δ I) β = Xᵀ y`
        with `X` and `y` standardized, recording all parameters.
        """

        if self.count == 0:
            print("alpha_model: no samples to fit", file=sys.stderr)
            return

        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        var_x = self.sum_xx / self.count - np.outer(mean_x, mean_x)
        var_y = self.sum_yy / self.count - mean_y * mean_y
        cov_xy = self.sum_xy / self.count - mean_x * mean_y
        std_x = np.sqrt(np.diag(var_x))
        std_x = np.where(std_x > 0.0, std_x, 1.0)
        std_y = np.sqrt(var_y)
        std_y = std_y if std_y > 0.0 else 1.0
        sum_standardized_xx = var_x * self.count / np.outer(std_x, std_x)
        sum_standardized_xy = cov_xy * self.count / (std_x * std_y)

        try:
            k = self.sum_x.shape[0]
            self.beta = np.linalg.solve(
                sum_standardized_xx
                + self.count * delta * np.eye(k),  # invertible with δ > 0
                sum_standardized_xy,
            )
            self.mean_x = mean_x
            self.mean_y = mean_y
            self.std_x = std_x
            self.std_y = std_y

        except np.linalg.LinAlgError:
            print("alpha_model: singular matrix", file=sys.stderr)

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """Predicts `ys` from `xs` using the fitted parameters."""

        m = xs.shape[0]
        assert xs.shape == (m, self.sum_x.shape[0]) and np.isfinite(xs).all()

        standardized_xs = (xs - self.mean_x) / self.std_x
        standardized_ys = standardized_xs @ self.beta
        return standardized_ys * self.std_y + self.mean_y


@dataclass(slots=True)
class AlphaModelState:
    target_offset: int
    min_periods: int
    ridge_l2: float

    # The last `target_offset + 1` feature cross-sections — just enough to
    # pair the oldest with the target that has now landed.
    pending: deque[np.ndarray]
    count: np.ndarray
    ridge: RidgeRegressor
    out_mu: np.ndarray


class AlphaModel:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = np.ndarray
    type Context = int
    type State = AlphaModelState

    def __init__(
        self,
        target_offset: int,
        min_periods: int,
        ridge_l2: float,
    ) -> None:
        assert target_offset >= 0, "alpha_model: target_offset must be non-negative"
        assert min_periods > 0, "alpha_model: min_periods must be positive"
        assert ridge_l2 > 0.0, "alpha_model: ridge_l2 must be positive"

        self.target_offset = target_offset
        self.min_periods = min_periods
        self.ridge_l2 = ridge_l2

    def init(self, inputs: Inputs) -> State:
        sample_signal, features, target, rebalance_signal = inputs
        n, k = features.shape
        assert target.shape == (n,)

        return AlphaModelState(
            target_offset=self.target_offset,
            min_periods=self.min_periods,
            ridge_l2=self.ridge_l2,
            pending=deque(maxlen=self.target_offset + 1),
            count=np.zeros((n,), dtype=np.int32),
            ridge=RidgeRegressor(k),
            out_mu=np.full((n,), np.nan),
        )

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return state.out_mu

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        sample_signal, features, target, rebalance_signal = inputs
        n, k = features.shape
        assert target.shape == (n,)

        if sample_signal:
            state.pending.append(features)
            if len(state.pending) == state.target_offset + 1:
                # Adds one pair to the training set (pooled).
                past_features = state.pending.popleft()
                valid = np.isfinite(past_features).all(axis=1) & np.isfinite(target)
                state.count += valid.astype(np.int32)
                state.ridge.add_samples(past_features[valid], target[valid])

        if rebalance_signal:
            # Predict `target_offset` periods ahead using the most recent features.
            valid = np.isfinite(features).all(axis=1) & (
                state.count >= state.min_periods
            )
            state.ridge.fit(state.ridge_l2)
            state.out_mu[valid] = state.ridge.predict(features[valid])

        return state.out_mu


def build(**kwargs) -> AlphaModel:
    return AlphaModel(**kwargs)
