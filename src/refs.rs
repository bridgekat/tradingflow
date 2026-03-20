use std::any::Any;

use super::observable::Observable;
use super::series::Series;

/// A permitted array scalar type.
pub trait Scalar: Copy + Send + Default + 'static {}

macro_rules! impl_scalar {
    ($($T:ty),+ $(,)?) => { $(impl Scalar for $T {})+ };
}

impl_scalar!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

/// A permitted input type.
///
/// Implemented by [`Observable`] and [`Series`].
pub trait Input {
    /// The scalar type of the input array.
    type Scalar: Scalar;

    /// Reconstruct a typed reference from a type-erased reference.
    fn from_erased(any: &dyn Any) -> &Self;

    /// The shape of the input array.
    fn shape(&self) -> &[usize];
}

/// A permitted output type.
///
/// Implemented by [`Observable`].
pub trait Output {
    /// The scalar type of the output array.
    type Scalar: Scalar;

    /// Reconstruct a typed reference from a type-erased reference.
    fn from_erased(any: &mut dyn Any) -> &mut Self;

    /// The shape of the output array.
    fn shape(&self) -> &[usize];
}

/// A collection of permitted input types.
///
/// Implemented by [`Input`], as well as slices and tuples containing
/// [`Input`]s.
pub trait Inputs {
    /// The collection of references.
    type Refs<'a>;

    /// Reconstruct typed references from type-erased references.
    fn from_erased<'a>(anys: Box<[&'a dyn Any]>) -> Self::Refs<'a>;

    /// The shapes of the input arrays.
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]>;
}

/// A collection of permitted output types.
///
/// Implemented by [`Output`], as well as slices and tuples containing
/// [`Output`]s.
pub trait Outputs {
    /// The collection of references.
    type Refs<'a>;

    /// The collection of mutable references.
    type RefMuts<'a>;

    /// Reconstruct typed references from type-erased references.
    fn from_erased<'a>(anys: Box<[&'a mut dyn Any]>) -> Self::RefMuts<'a>;

    /// The shapes of the output arrays.
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]>;
}

impl<T: Scalar> Input for Observable<T> {
    type Scalar = T;

    #[inline(always)]
    fn from_erased(any: &dyn Any) -> &Self {
        any.downcast_ref().unwrap()
    }

    #[inline(always)]
    fn shape(&self) -> &[usize] {
        Observable::shape(self)
    }
}

impl<T: Scalar> Input for Series<T> {
    type Scalar = T;

    #[inline(always)]
    fn from_erased(any: &dyn Any) -> &Self {
        any.downcast_ref().unwrap()
    }

    #[inline(always)]
    fn shape(&self) -> &[usize] {
        Series::shape(self)
    }
}

impl<T: Scalar> Output for Observable<T> {
    type Scalar = T;

    #[inline(always)]
    fn from_erased(any: &mut dyn Any) -> &mut Self {
        any.downcast_mut::<Self>().unwrap()
    }

    #[inline(always)]
    fn shape(&self) -> &[usize] {
        Observable::shape(self)
    }
}

impl<T: Input + 'static> Inputs for T {
    type Refs<'a> = &'a T;

    #[inline(always)]
    fn from_erased<'a>(anys: Box<[&'a dyn Any]>) -> Self::Refs<'a> {
        T::from_erased(anys[0])
    }

    #[inline(always)]
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
        Box::new([refs.shape()])
    }
}

impl<T: Input + 'static> Inputs for [T] {
    type Refs<'a> = Box<[&'a T]>;

    #[inline(always)]
    fn from_erased<'a>(anys: Box<[&'a dyn Any]>) -> Self::Refs<'a> {
        anys.into_iter().map(T::from_erased).collect()
    }

    #[inline(always)]
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
        refs.into_iter().map(T::shape).collect()
    }
}

macro_rules! impl_input_refs_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Input + 'static),+> Inputs for ($($T,)+) {
            type Refs<'a> = ($(&'a $T,)+);

            #[inline(always)]
            fn from_erased<'a>(anys: Box<[&'a dyn Any]>) -> Self::Refs<'a> {
                let mut it = anys.into_iter();
                ($($T::from_erased(it.next().unwrap()),)+)
            }

            #[inline(always)]
            fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
                Box::new([$(refs.$idx.shape()),+])
            }
        }
    };
}

impl_input_refs_for_tuple!(0: A);
impl_input_refs_for_tuple!(0: A, 1: B);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

impl<T: Output + 'static> Outputs for T {
    type Refs<'a> = &'a T;
    type RefMuts<'a> = &'a mut T;

    #[inline(always)]
    fn from_erased<'a>(anys: Box<[&'a mut dyn Any]>) -> Self::RefMuts<'a> {
        T::from_erased(anys[0])
    }

    #[inline(always)]
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
        Box::new([refs.shape()])
    }
}

impl<T: Output + 'static> Outputs for [T] {
    type Refs<'a> = Box<[&'a T]>;
    type RefMuts<'a> = Box<[&'a mut T]>;

    #[inline(always)]
    fn from_erased<'a>(anys: Box<[&'a mut dyn Any]>) -> Self::RefMuts<'a> {
        anys.into_iter().map(T::from_erased).collect()
    }

    #[inline(always)]
    fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
        refs.into_iter().map(T::shape).collect()
    }
}

macro_rules! impl_output_refs_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Output + 'static),+> Outputs for ($($T,)+) {
            type Refs<'a> = ($(&'a $T,)+);
            type RefMuts<'a> = ($(&'a mut $T,)+);

            #[inline(always)]
            fn from_erased<'a>(anys: Box<[&'a mut dyn Any]>) -> Self::RefMuts<'a> {
                let mut it = anys.into_iter();
                ($($T::from_erased(it.next().unwrap()),)+)
            }

            #[inline(always)]
            fn shapes<'a>(refs: Self::Refs<'a>) -> Box<[&'a [usize]]> {
                Box::new([$(refs.$idx.shape()),+])
            }
        }
    };
}

impl_output_refs_for_tuple!(0: A);
impl_output_refs_for_tuple!(0: A, 1: B);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_output_refs_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
