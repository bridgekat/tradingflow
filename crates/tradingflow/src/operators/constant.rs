use crate::data::{Array, Scalar, Series};
use crate::graph::typed::Source;
use crate::ports::{ArrayPass, SeriesPass};

/// A constant [`Array`] cell.
pub fn array_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Array<T, N>,
) -> Source<ArrayPass<T, N>, C> {
    Source::new(initial)
}

/// A constant [`Series`] cell.
pub fn series_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Series<T, N>,
) -> Source<SeriesPass<T, N>, C> {
    Source::new(initial)
}
