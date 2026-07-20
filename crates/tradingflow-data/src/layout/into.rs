use super::{Slice, SliceReshape};

pub trait IntoSlices<const N: usize> {
    fn into_slices(self) -> [Slice; N];
}

pub trait IntoSliceReshapes<const N: usize> {
    fn into_slice_reshapes(self) -> [SliceReshape; N];
}

impl<T: Into<Slice>, const N: usize> IntoSlices<N> for [T; N] {
    fn into_slices(self) -> [Slice; N] {
        self.map(Into::into)
    }
}

impl<T: Into<SliceReshape>, const N: usize> IntoSliceReshapes<N> for [T; N] {
    fn into_slice_reshapes(self) -> [SliceReshape; N] {
        self.map(Into::into)
    }
}

macro_rules! impl_into_specifiers_for_tuple {
    ($N:literal; $($idx:tt: $T:ident),*) => {
        impl<$($T: Into<Slice>,)*> IntoSlices<{ $N }> for ($($T,)*) {
            fn into_slices(self) -> [Slice; $N] {
                [$(self.$idx.into(),)*]
            }
        }

        impl<$($T: Into<SliceReshape>,)*> IntoSliceReshapes<{ $N }> for ($($T,)*) {
            fn into_slice_reshapes(self) -> [SliceReshape; $N] {
                [$(self.$idx.into(),)*]
            }
        }
    };
}

impl_into_specifiers_for_tuple!(0; );
impl_into_specifiers_for_tuple!(1; 0: A);
impl_into_specifiers_for_tuple!(2; 0: A, 1: B);
impl_into_specifiers_for_tuple!(3; 0: A, 1: B, 2: C);
impl_into_specifiers_for_tuple!(4; 0: A, 1: B, 2: C, 3: D);
impl_into_specifiers_for_tuple!(5; 0: A, 1: B, 2: C, 3: D, 4: E);
impl_into_specifiers_for_tuple!(6; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_into_specifiers_for_tuple!(7; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_into_specifiers_for_tuple!(8; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_into_specifiers_for_tuple!(9; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_into_specifiers_for_tuple!(10; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_into_specifiers_for_tuple!(11; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_into_specifiers_for_tuple!(12; 0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
