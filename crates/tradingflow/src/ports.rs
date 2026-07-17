use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Scalar, Series, SeriesView};
use crate::graph::typed::{Interface, Value, ViewPort, ViewPorts};

/// The [`Value`] policy which passes borrowed [`ArrayView<'a, T, N>`]
/// for [`Array<T, N>`] across graph interfaces.
pub struct ArrayValue<T, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Value for ArrayValue<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
    type Owned = Array<T, N>;

    #[inline(always)]
    fn borrow(owned: &Array<T, N>) -> ArrayView<'_, T, N> {
        owned.view()
    }
}

/// The [`Value`] policy which passes borrowed [`SeriesView<'a, T, N>`]
/// for [`Series<T, N>`] across graph interfaces.
pub struct SeriesValue<T, const N: usize>(PhantomData<T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]`,
// which is covariant in `'a`.
unsafe impl<T: Scalar, const N: usize> Value for SeriesValue<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
    type Owned = Series<T, N>;

    #[inline(always)]
    fn borrow(owned: &Series<T, N>) -> SeriesView<'_, T, N> {
        owned.view()
    }
}

/// A single port carrying a strided [`ArrayView<T, N>`](crate::data::ArrayView)
/// by value — the array-shaped edge currency.
pub type ArrayPort<T, const N: usize> = ViewPort<ArrayValue<T, N>>;

/// A runtime-length group of [`ArrayPort`]s, payload `(&[bool],
/// &[ArrayView<T, N>])` — wires against a slice of independent [`ArrayPort`]
/// producer handles (`&[PortHandle<ArrayPort<T, N>>]`), no bridging adapters.
pub type ArrayPorts<T, const N: usize> = ViewPorts<ArrayValue<T, N>>;

/// A single port carrying a [`SeriesView<T, N>`](crate::data::SeriesView)
/// (recorded history window) by value — the [`Series`] edge currency.
pub type SeriesPort<T, const N: usize> = ViewPort<SeriesValue<T, N>>;

/// A runtime-length group of [`SeriesPort`]s, payload `(&[bool],
/// &[SeriesView<T, N>])`.
pub type SeriesPorts<T, const N: usize> = ViewPorts<SeriesValue<T, N>>;

/// Maps an [`Interface`] into its values-only type tree,
/// for use by the map operators.
pub trait StripNotify: Interface {
    type Plain<'a>: Copy;
    fn plain<'a>(values: Self::Values<'a>) -> Self::Plain<'a>;
}

impl<V: Value> StripNotify for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = V::View<'a>;

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl<V: Value> StripNotify for ViewPorts<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = &'a [V::View<'a>];

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl StripNotify for () {
    type Plain<'a> = ();

    #[inline(always)]
    fn plain<'a>(_: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {}
}

macro_rules! impl_strip_notify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: StripNotify,)+> StripNotify for ($($T,)+) {
            type Plain<'a> = ($($T::Plain<'a>,)+);

            #[inline(always)]
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
