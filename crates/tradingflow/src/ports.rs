//! Segment interfaces for arrays and series.

use num_traits::Float;
use std::marker::PhantomData;

use crate::data::{ArrayView, Scalar, SeriesView};
use crate::graph::{Pass, Port, PortHandle, Ports, Val};

/// The [`Pass`] policy which passes borrowed [`ArrayView<'a, T, N>`]
/// for [`Array<T, N>`](crate::data::Array) across graph interfaces.
pub struct ArrayPass<T: Scalar, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for ArrayPass<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
}

/// The [`Pass`] policy which passes borrowed [`SeriesView<'a, T, N>`]
/// for [`Series<T, N>`](crate::data::Series) across graph interfaces.
pub struct SeriesPass<T: Scalar, const N: usize>(PhantomData<T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Pass for SeriesPass<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
}

// /// A marker around [`ArrayPass<T, N>`] indicating an event array rather than
// /// a state array, i.e. an array whose non-NaN values are one-time events.
// pub struct EventArrayPass<T: Scalar + Float, const N: usize>(ArrayPass<T, N>);

// // SAFETY: same as above.
// unsafe impl<T: Scalar + Float, const N: usize> Pass for EventArrayPass<T, N> {
//     type View<'a> = ArrayView<'a, T, N>;
// }

/// A single port carrying a flag.
pub type ClockPort = Port<Val<bool>>;

/// A single port carrying an array by [`ArrayView<T, N>`].
pub type ArrayPort<T, const N: usize> = Port<ArrayPass<T, N>>;

/// A single port carrying a series by [`SeriesView<T, N>`].
pub type SeriesPort<T, const N: usize> = Port<SeriesPass<T, N>>;

// /// A single port carrying an event array by [`ArrayView<T, N>`].
// pub type EventArrayPort<T, const N: usize> = Port<EventArrayPass<T, N>>;

/// A runtime-length group of [`ClockPort`]s.
pub type ClockPorts = Ports<Val<bool>>;

/// A runtime-length group of [`ArrayPort`]s.
pub type ArrayPorts<T, const N: usize> = Ports<ArrayPass<T, N>>;

/// A runtime-length group of [`SeriesPort`]s.
pub type SeriesPorts<T, const N: usize> = Ports<SeriesPass<T, N>>;

// /// A runtime-length group of [`EventArrayPort`]s.
// pub type EventArrayPorts<T, const N: usize> = Ports<EventArrayPass<T, N>>;

/// A handle to a single [`ClockPort`].
pub type ClockPortHandle = PortHandle<Val<bool>>;

/// A handle to a single [`ArrayPort`].
pub type ArrayPortHandle<T, const N: usize> = PortHandle<ArrayPass<T, N>>;

/// A handle to a single [`SeriesPort`].
pub type SeriesPortHandle<T, const N: usize> = PortHandle<SeriesPass<T, N>>;

// /// A handle to a single [`EventArrayPort`].
// pub type EventArrayPortHandle<T, const N: usize> = PortHandle<EventArrayPass<T, N>>;

/// Whether an event array contains any events.
pub fn is_eventful<'a, T: Scalar + Float, const N: usize>(view: ArrayView<'a, T, N>) -> bool {
    if view.data().len() == 1 && view.data()[0].is_nan() {
        return false;
    }
    view.iter().any(|x| !x.is_nan())
}
