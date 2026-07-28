//! Segment interfaces for arrays and series.

use std::marker::PhantomData;

use crate::data::{ArrayView, Scalar, SeriesView};
use crate::graph::{Pass, Port, PortHandle, Ports};

/// The [`Pass`] protocol which passes borrowed [`ArrayView<'a, T, N>`]
/// for [`Array<T, N>`](crate::data::Array) across graph interfaces.
pub struct ArrayPass<T: Scalar, const N: usize>(PhantomData<fn() -> T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for ArrayPass<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
}

/// The [`Pass`] protocol which passes borrowed [`SeriesView<'a, T, N>`]
/// for [`Series<T, N>`](crate::data::Series) across graph interfaces.
pub struct SeriesPass<T: Scalar, const N: usize>(PhantomData<fn() -> T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for SeriesPass<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
}

/// A marker around [`ArrayPass<bool, N>`] indicating a clock signal array.
///
/// Clock signals should always be reset to `false` (by their producer nodes)
/// at the end of each generation.
pub struct ClockArrayPass<const N: usize>(ArrayPass<bool, N>);

pub type ClockPass = ClockArrayPass<0>;

// SAFETY: same as above.
unsafe impl<const N: usize> Pass for ClockArrayPass<N> {
    type View<'a> = ArrayView<'a, bool, N>;
}

/// A single port carrying an array by [`ArrayView<T, N>`].
pub type ArrayPort<T, const N: usize> = Port<ArrayPass<T, N>>;

/// A single port carrying a series by [`SeriesView<T, N>`].
pub type SeriesPort<T, const N: usize> = Port<SeriesPass<T, N>>;

/// A single port carrying a clock signal array.
pub type ClockArrayPort<const N: usize> = Port<ClockArrayPass<N>>;

pub type ClockPort = ClockArrayPort<0>;

/// A runtime-length group of [`ArrayPort`]s.
pub type ArrayPorts<T, const N: usize> = Ports<ArrayPass<T, N>>;

/// A runtime-length group of [`SeriesPort`]s.
pub type SeriesPorts<T, const N: usize> = Ports<SeriesPass<T, N>>;

/// A runtime-length group of [`ClockArrayPort`]s.
pub type ClockArrayPorts<const N: usize> = Ports<ClockArrayPass<N>>;

pub type ClockPorts = ClockArrayPorts<0>;

/// A handle to a single [`ArrayPort`].
pub type ArrayPortHandle<T, const N: usize> = PortHandle<ArrayPass<T, N>>;

/// A handle to a single [`SeriesPort`].
pub type SeriesPortHandle<T, const N: usize> = PortHandle<SeriesPass<T, N>>;

/// A handle to a single [`ClockArrayPort`].
pub type ClockArrayPortHandle<const N: usize> = PortHandle<ClockArrayPass<N>>;

pub type ClockPortHandle = ClockArrayPortHandle<0>;
