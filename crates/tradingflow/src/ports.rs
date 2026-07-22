//! Segment interfaces for arrays and series.

use std::marker::PhantomData;

use crate::data::{ArrayView, Scalar, SeriesView};
use crate::graph::{Interface, Pass, Port, PortHandle, Ports, Val};

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

/// Maps an [`Interface`] into its values-only type tree,
/// for use by the map operators.
pub trait StripNotify: Interface {
    type Plain<'a>: Copy;
    fn plain<'a>(values: Self::Values<'a>) -> Self::Plain<'a>;
}

impl<V: Pass> StripNotify for Port<V> {
    type Plain<'a> = V::View<'a>;

    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl<V: Pass> StripNotify for Ports<V> {
    type Plain<'a> = &'a [V::View<'a>];

    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl StripNotify for () {
    type Plain<'a> = ();

    fn plain<'a>(_: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {}
}

macro_rules! impl_strip_notify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: StripNotify,)+> StripNotify for ($($T,)+) {
            type Plain<'a> = ($($T::Plain<'a>,)+);

            fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
                ( $( <$T as StripNotify>::plain(values.$idx), )+ )
            }
        }
    };
}

impl_strip_notify_for_tuple!(0: A);
impl_strip_notify_for_tuple!(0: A, 1: B);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
