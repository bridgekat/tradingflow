use crate::data::{Array, ArrayView, Instant, Scalar, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Emits the recorded-history element from `offset` steps ago (else `fill`),
/// as a rank-`N` view of its homed buffer. Consumes a [`SeriesPort`] window;
/// the look-back is relative to the window's newest row, so a
/// retention-bounded record works as long as the bound covers `offset + 1`
/// rows.
#[derive(Clone)]
pub struct Lag<T: Scalar, const N: usize> {
    offset: usize,
    fill: T,
}

impl<T: Scalar, const N: usize> Lag<T, N> {
    pub fn new(offset: usize, fill: T) -> Self {
        Self { offset, fill }
    }
}

/// Runtime state for [`Lag`]: the configuration plus the output buffer (sized
/// and seeded with the fill value in `init`).
pub struct LagState<T: Scalar, const N: usize> {
    offset: usize,
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Lag<T, N> {
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = LagState<T, N>;

    fn init(self, (_, series): (bool, SeriesView<'_, T, N>)) -> Self::State {
        LagState {
            offset: self.offset,
            out: Array::full(series.extents(), self.fill.clone()),
            fill: self.fill,
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        let len = series.len();
        if len > state.offset {
            let (_, lagged) = series.at(len - 1 - state.offset).unwrap();
            state.out.assign(lagged);
        } else {
            state.out.data_mut().fill(state.fill.clone());
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, SeriesView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// The value from `offset` ticks ago in a recorded [`Series`](tradingflow_data::Series), `fill` until it
/// exists — the primitive behind the self-recording [`lag`](crate::operators::formula::lag).
/// (Named `_series` because `lag` is taken by its live-array counterpart.)
pub fn lag_series<T: Scalar, const N: usize>(offset: usize, fill: T) -> Lag<T, N> {
    Lag::new(offset, fill)
}
