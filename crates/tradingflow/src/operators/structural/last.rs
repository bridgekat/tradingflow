//! `Last` — most recent element of a Series as an Array.

use crate::data::{Array, ArrayView, Instant, Scalar, SeriesView};
use crate::graph::typed::Operator;
use crate::ports::{ArrayPort, SeriesPort};

/// Extracts the most recent element of a `Series<T, N>` as a rank-`N`
/// [`ArrayView`], substituting `fill` when the series is empty.
pub struct Last<T: Scalar, const N: usize> {
    fill: T,
}

impl<T: Scalar, const N: usize> Last<T, N> {
    pub fn new(fill: T) -> Self {
        Self { fill }
    }
}

/// Runtime state for [`Last`]: the fill value plus the output buffer.
pub struct LastState<T: Scalar, const N: usize> {
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Last<T, N> {
    type Inputs = SeriesPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = LastState<T, N>;

    fn init(self, (_, series): (bool, SeriesView<'_, T, N>)) -> Self::State {
        let mut out = Array::full(series.extents(), self.fill.clone());
        if !series.is_empty() {
            out.assign(series.at(series.len() - 1).unwrap().1);
        }
        LastState {
            fill: self.fill,
            out,
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, SeriesView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if series.is_empty() {
            let fill = state.fill.clone();
            state.out.data_mut().fill(fill);
        } else {
            state.out.assign(series.at(series.len() - 1).unwrap().1);
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

/// The most recent element of a [`Series`](crate::data::Series) as an array
/// view, `fill` when empty.
pub fn last<T: Scalar, const N: usize>(fill: T) -> Last<T, N> {
    Last::new(fill)
}
