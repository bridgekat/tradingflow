//! The [`Scalar`] trait.

/// A permitted array scalar type. Can be extended.
pub trait Scalar: Clone + Default + Send + Sync + 'static {}

impl Scalar for () {}

impl Scalar for bool {}

impl Scalar for i8 {}

impl Scalar for i16 {}

impl Scalar for i32 {}

impl Scalar for i64 {}

impl Scalar for u8 {}

impl Scalar for u16 {}

impl Scalar for u32 {}

impl Scalar for u64 {}

impl Scalar for f32 {}

impl Scalar for f64 {}

impl<T: Scalar> Scalar for Option<T> {}

macro_rules! impl_scalar_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: Scalar),+> Scalar for ($($T,)+) {}
    };
}

impl_scalar_for_tuple!(A);
impl_scalar_for_tuple!(A, B);
impl_scalar_for_tuple!(A, B, C);
impl_scalar_for_tuple!(A, B, C, D);
impl_scalar_for_tuple!(A, B, C, D, E);
impl_scalar_for_tuple!(A, B, C, D, E, F);
impl_scalar_for_tuple!(A, B, C, D, E, F, G);
impl_scalar_for_tuple!(A, B, C, D, E, F, G, H);
impl_scalar_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_scalar_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_scalar_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_scalar_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
