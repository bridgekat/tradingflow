from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class InformationCoefficientState:
    predict: np.ndarray | None = None
    sum: float = 0.0
    count: int = 0
    out: float = field(default=float("nan"))


class InformationCoefficient:
    type Inputs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    type Outputs = tuple[bool, float]
    type Context = int
    type State = InformationCoefficientState

    def __init__(self):
        pass

    def init(self, _: Inputs) -> State:
        return InformationCoefficientState()

    @staticmethod
    def reset(_: Inputs, state: State) -> Outputs:
        return (False, state.out)

    @staticmethod
    def compute(inputs: Inputs, state: State, _: Context) -> Outputs:
        predict_signal, predict, target_signal, target = inputs

        if target_signal and state.predict is not None:
            # Fold one cross-sectional correlation per sampling period.
            valid = np.isfinite(state.predict) & np.isfinite(target)
            s, r = state.predict[valid], target[valid]
            if len(s) >= 2:
                state.sum += float(np.corrcoef(s, r)[0, 1])
            state.count += 1

        if predict_signal:
            # A new prediction closes the open evaluation period and opens the
            # next. The first one has no period to close.
            emit = state.predict is not None
            if emit:
                state.out = state.sum / max(state.count, 1)
            state.predict = predict
            state.sum = 0.0
            state.count = 0
            return (emit, state.out)
        else:
            return (False, state.out)


__op__ = InformationCoefficient()
