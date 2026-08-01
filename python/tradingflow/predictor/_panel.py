"""Recording harness shared by the windowed mean and variance predictors.

Both fit the same way — accumulate a window of cross-sections, refit on a
cadence, emit one prediction per rebalance — and differ in the shape of what
they emit, how a per-stock mask scatters into it, and whether they read the
feature panel at all. Those are the hooks, and they live on the *state* rather
than on the segment, so `reset` and `compute` need no `self`.
"""

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Callable

import numpy as np


@dataclass(slots=True)
class PanelState:
    """Everything a windowed predictor carries between generations.

    This mirrors the Rust `Segment::State`. There, `init` takes `self` *by
    value*, so the build configuration is moved into the state and the segment
    ceases to exist — which is why `reset` and `compute` are free functions of
    `(inputs, state)`. Python cannot enforce that, but it can follow it: `init`
    is the only method that reads the segment, everything it needs is copied
    here, and the subclass hooks hang off the state too.

    It matters beyond tidiness. A module binding `__op__` is imported once, so
    *every node loading it shares one segment instance* — seven of them, in the
    `covariance_gmv` example. Anything left on `self` would be shared across all
    of them; anything in the state is per-node by construction. `slots=True`
    turns a stray attribute into an `AttributeError` rather than a silent
    cross-node alias.
    """

    #: `(x, y) -> params` over the training block, and
    #: `(features, params) -> predictions` over the active cross-section.
    fit: Callable
    predict: Callable
    target_offset: int
    refit_every: int
    max_periods: int | None
    min_periods: int | None
    universe_size: int | None
    #: Whether the fit needs a *window* of past feature cross-sections.
    #:
    #: This is the expensive question, and it is a property of the model
    #: rather than of what it predicts — hence a flag on both bases rather
    #: than a fact about mean versus variance. The current cross-section
    #: arrives as an input on every generation and never has to be retained;
    #: only a fit that reaches backwards pays for history, and on a wide panel
    #: that is the difference between megabytes and gigabytes.
    #:
    #: When false the window is never accumulated, `fit` is handed `x=None`,
    #: coverage counts come from the target alone, and no stock is excluded
    #: for having unusable features — a model that never looks at them has no
    #: grounds to.
    retain_features: bool
    #: Recorded `(N, F)` feature cross-sections, newest last. Stays empty
    #: unless [`retain_features`][PanelState.retain_features].
    features: deque
    #: Recorded `(N,)` target cross-sections, newest last.
    target: deque
    #: The retained prediction, re-emitted on every non-rebalance tick.
    out: np.ndarray
    params: object = None
    fitted: bool = False
    rebalances: int = 0

    @staticmethod
    def empty(n: int) -> np.ndarray:
        """An all-`NaN` output for a cross-section of `n` stocks."""
        raise NotImplementedError

    @staticmethod
    def scatter(out: np.ndarray, mask: np.ndarray, values: np.ndarray) -> None:
        """Places the masked stocks' `values` into `out`."""
        raise NotImplementedError


def window(dq: deque, start: int, stop: int) -> list:
    """The `[start, stop)` slice of a deque, as a list of cross-sections."""
    return list(islice(dq, max(start, 0), max(stop, 0)))


class PanelPredictor:
    r"""Windowed panel predictor: record, refit on a cadence, predict.

    On each sampling tick one `(features, target)` cross-section pair is
    appended to a bounded window. On each rebalance tick the window is
    flattened into a training block — `features[i]` paired with
    `target[i + target_offset]` — the model is refit if the cadence is due, and
    a prediction is emitted for every stock that passes the mask.

    A stock is in the mask when it is in the universe, has at least
    `min_periods` valid observations in the window, and — for a model that
    reads features — has finite current features. Everything else stays at
    `NaN`, which is how downstream portfolios and metrics recognise a stock the
    model could not price.
    """

    #: The [`PanelState`] subclass this predictor builds, carrying the output
    #: shape, the scatter rule and whether features are read.
    state_type: type[PanelState] = PanelState

    def __init__(
        self,
        *,
        fit,
        predict,
        retain_features: bool = True,
        target_offset: int = 0,
        refit_every: int = 1,
        max_periods: int | None = None,
        min_periods: int | None = None,
        universe_size: int | None = None,
    ) -> None:
        assert target_offset >= 0, "target_offset must be non-negative"
        assert refit_every >= 1, "refit_every must be >= 1"
        assert max_periods is None or max_periods >= 1, "max_periods must be >= 1"
        assert min_periods is None or min_periods >= 1, "min_periods must be >= 1"

        self.fit = fit
        self.predict = predict
        self.retain_features = retain_features
        self.target_offset = int(target_offset)
        self.refit_every = int(refit_every)
        self.max_periods = max_periods
        self.min_periods = min_periods
        self.universe_size = universe_size

    def init(self, inputs) -> PanelState:
        *_, universe = inputs
        # A bounded deque does the trimming: forming `max_periods` pairs needs
        # `max_periods + target_offset` cross-sections once the forward offset
        # is accounted for, and the oldest fall off the left as they age out.
        maxlen = None if self.max_periods is None else self.max_periods + self.target_offset
        return self.state_type(
            fit=self.fit,
            predict=self.predict,
            retain_features=self.retain_features,
            target_offset=self.target_offset,
            refit_every=self.refit_every,
            max_periods=self.max_periods,
            min_periods=self.min_periods,
            universe_size=self.universe_size,
            features=deque(maxlen=maxlen),
            target=deque(maxlen=maxlen),
            out=self.state_type.empty(universe.shape[0]),
        )

    @staticmethod
    def reset(_, state: PanelState):
        return (False, state.out)

    @staticmethod
    def compute(inputs, state: PanelState, _):
        sample_signal, features, target, rebalance_signal, universe = inputs

        if sample_signal:
            state.target.append(target)
            if state.retain_features:
                state.features.append(features)

        if not rebalance_signal:
            return (False, state.out)

        n = universe.shape[0]
        # Emit on every rebalance whatever happens, so downstream metrics see
        # one prediction per period; an unfittable panel emits all-NaN.
        out = state.empty(n)
        if not state.target:
            state.out = out
            return (True, state.out)

        # Pairs run features[i] with target[i + target_offset]; the training
        # block is the last `n_use` of them, which — since the deques are
        # trimmed together — ends at the newest target.
        length = len(state.target)
        n_pair = max(0, length - state.target_offset)
        n_use = n_pair if state.max_periods is None else min(n_pair, state.max_periods)

        counts = np.zeros(n)
        x = y = None
        if n_use > 0:
            y = np.stack(window(state.target, length - n_use, length))  # (T, N)
            valid = np.isfinite(y)
            if state.retain_features:
                x = np.stack(window(state.features, n_pair - n_use, n_pair))  # (T, N, F)
                valid &= np.isfinite(x).all(axis=2)
            counts = valid.sum(axis=0)

        current = state.features[-1] if state.retain_features else None

        mask = universe > 0
        if state.universe_size is not None:
            assert int(mask.sum()) <= state.universe_size, (
                f"universe has {int(mask.sum())} nonzero entries, "
                f"exceeding universe_size={state.universe_size}"
            )
        if state.min_periods is not None:
            mask = mask & (counts >= state.min_periods)
        if current is not None:
            mask = mask & np.isfinite(current).all(axis=1)

        # Refit on the cadence, or whenever there is still nothing to predict
        # with; otherwise reuse the parameters from the last refit.
        refit = (not state.fitted) or (state.rebalances % state.refit_every == 0)
        state.rebalances += 1
        if refit and n_use > 0 and mask.any():
            state.params = state.fit(x[:, mask, :] if x is not None else None, y[:, mask])
            state.fitted = True

        if state.fitted and mask.any():
            active = current[mask] if current is not None else None
            state.scatter(out, mask, state.predict(active, state.params))
        state.out = out
        return (True, state.out)
