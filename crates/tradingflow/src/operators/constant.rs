use crate::data::{Array, Scalar, Series};
use crate::graph::typed::ViewSource;
use crate::ports::{ArrayValue, SeriesValue};

/// A constant [`Array`] cell.
pub fn array_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Array<T, N>,
) -> ViewSource<ArrayValue<T, N>, C> {
    ViewSource::new(initial)
}

/// A constant [`Series`] cell.
pub fn series_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Series<T, N>,
) -> ViewSource<SeriesValue<T, N>, C> {
    ViewSource::new(initial)
}
