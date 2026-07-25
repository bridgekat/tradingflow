//! Segment interfaces for arrays and series.

use std::marker::PhantomData;

use crate::data::{ArrayView, Scalar, SeriesView};
use crate::graph::{Pass, Port, PortHandle, Ports, Val};

/// The [`Pass`] policy which passes borrowed [`ArrayView<'a, T, N>`]
/// for [`Array<T, N>`](crate::data::Array) across graph interfaces.
pub struct ArrayPass<T, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for ArrayPass<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
}

/// The [`Pass`] policy which passes borrowed [`SeriesView<'a, T, N>`]
/// for [`Series<T, N>`](crate::data::Series) across graph interfaces.
pub struct SeriesPass<T, const N: usize>(PhantomData<T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for SeriesPass<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
}

/// A single port carrying no payload.
pub type UnitPort = Port<Val<()>>;

/// A single port carrying an array by
/// [`ArrayView<T, N>`](crate::data::ArrayView).
pub type ArrayPort<T, const N: usize> = Port<ArrayPass<T, N>>;

/// A single port carrying a series by
/// [`SeriesView<T, N>`](crate::data::SeriesView).
pub type SeriesPort<T, const N: usize> = Port<SeriesPass<T, N>>;

/// A runtime-length group of [`UnitPort`]s.
pub type UnitPorts = Ports<Val<()>>;

/// A runtime-length group of [`ArrayPort`]s.
pub type ArrayPorts<T, const N: usize> = Ports<ArrayPass<T, N>>;

/// A runtime-length group of [`SeriesPort`]s.
pub type SeriesPorts<T, const N: usize> = Ports<SeriesPass<T, N>>;

/// A handle to a single [`UnitPort`].
pub type UnitPortHandle = PortHandle<Val<()>>;

/// A handle to a single [`ArrayPort`].
pub type ArrayPortHandle<T, const N: usize> = PortHandle<ArrayPass<T, N>>;

/// A handle to a single [`SeriesPort`].
pub type SeriesPortHandle<T, const N: usize> = PortHandle<SeriesPass<T, N>>;

/// Inspects an array port, returning the array view if notified, or an array
/// view of the same extents filled with a static `none` value otherwise.
///
/// Useful for operators implementing event semantics.
pub fn event_or<'a, T: Scalar, const N: usize>(
    (notify, view): (bool, ArrayView<'a, T, N>),
    none: &'a T,
) -> ArrayView<'a, T, N> {
    if notify {
        view
    } else {
        ArrayView::full(view.extents(), none)
    }
}
