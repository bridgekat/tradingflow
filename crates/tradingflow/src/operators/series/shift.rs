use super::view::DeriveView;
use crate::data::{Instant, Scalar, series};
use crate::graph::Segment;
use crate::ports::SeriesPort;

/// Shifts the index of a series view by `n` periods: [`series::shift`].
///
/// Retains only the elements retained in the input series.
pub fn shift<T: Scalar, const N: usize>(
    n: isize,
) -> impl Segment<Inputs = SeriesPort<T, N>, Outputs = SeriesPort<T, N>, Context = Instant> {
    DeriveView::new(move |a| series::shift(a, n))
}
