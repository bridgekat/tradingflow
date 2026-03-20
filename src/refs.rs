use super::observable::Observable;
use super::series::Series;

/// A reference to a permitted input type.
///
/// Implemented by references to [`Observable`] and [`Series`].
pub trait InputRef<'a> {
    /// The scalar type of the input array.
    type Scalar: Copy;

    /// Reconstruct a typed reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to the correct concrete type and be valid for `'a`.
    unsafe fn from_raw(ptr: *const u8) -> Self;

    /// The shape of the input array.
    fn shape(self) -> &'a [usize];
}

/// A mutable reference to a permitted output type.
///
/// Implemented by mutable references to [`Observable`].
pub trait OutputRef<'a> {
    /// The scalar type of the output array.
    type Scalar: Copy;

    /// Reconstruct a typed mutable reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to the correct concrete type and be valid for `'a`.
    unsafe fn from_raw(ptr: *mut u8) -> Self;

    /// The shape of the output array.
    fn shape(self) -> &'a [usize];
}

/// A collection of references to permitted input types.
///
/// Implemented by [`InputRef`], as well as slices and tuples
/// containing [`InputRef`]s.
pub trait InputRefs<'a> {
    /// Reconstruct typed references from raw pointers.
    ///
    /// # Safety
    ///
    /// `ptrs` must contain pointers to correct concrete types and be valid for `'a`.
    unsafe fn from_raw(ptrs: &[*const u8]) -> Self;

    /// The shapes of the input arrays.
    fn shapes(self) -> Box<[&'a [usize]]>;
}

/// A collection of references to permitted output types.
///
/// Implemented by [`OutputRef`], as well as slices and tuples
/// containing [`OutputRef`]s.
pub trait OutputRefs<'a> {
    /// Reconstruct typed mutable references from raw pointers.
    ///
    /// # Safety
    ///
    /// `ptrs` must contain pointers to correct concrete types and be valid for `'a`.
    unsafe fn from_raw(ptrs: &[*mut u8]) -> Self;

    /// The shapes of the output arrays.
    fn shapes(self) -> Box<[&'a [usize]]>;
}

impl<'a, T: Copy> InputRef<'a> for &'a Observable<T> {
    type Scalar = T;

    #[inline(always)]
    unsafe fn from_raw(ptr: *const u8) -> Self {
        unsafe { &*(ptr as *const Observable<T>) }
    }

    #[inline(always)]
    fn shape(self) -> &'a [usize] {
        Observable::shape(self)
    }
}

impl<'a, T: Copy> InputRef<'a> for &'a Series<T> {
    type Scalar = T;

    #[inline(always)]
    unsafe fn from_raw(ptr: *const u8) -> Self {
        unsafe { &*(ptr as *const Series<T>) }
    }

    #[inline(always)]
    fn shape(self) -> &'a [usize] {
        Series::shape(self)
    }
}

impl<'a, T: Copy> OutputRef<'a> for &'a mut Observable<T> {
    type Scalar = T;

    #[inline(always)]
    unsafe fn from_raw(ptr: *mut u8) -> Self {
        unsafe { &mut *(ptr as *mut Observable<T>) }
    }

    #[inline(always)]
    fn shape(self) -> &'a [usize] {
        Observable::shape(self)
    }
}

impl<'a, R: InputRef<'a>> InputRefs<'a> for R {
    #[inline(always)]
    unsafe fn from_raw(ptrs: &[*const u8]) -> Self {
        unsafe { R::from_raw(ptrs[0]) }
    }

    #[inline(always)]
    fn shapes(self) -> Box<[&'a [usize]]> {
        Box::new([self.shape()])
    }
}

impl<'a, R: InputRef<'a>> InputRefs<'a> for Box<[R]> {
    #[inline(always)]
    unsafe fn from_raw(ptrs: &[*const u8]) -> Self {
        unsafe { ptrs.iter().map(|&ptr| R::from_raw(ptr)).collect() }
    }

    #[inline(always)]
    fn shapes(self) -> Box<[&'a [usize]]> {
        self.into_iter().map(|r| r.shape()).collect()
    }
}

macro_rules! impl_input_refs_for_tuple {
    ($($idx:tt: $R:ident),+ $(,)?) => {
        impl<'a, $($R: InputRef<'a>),+> InputRefs<'a> for ($($R,)+) {
            #[inline(always)]
            unsafe fn from_raw(ptrs: &[*const u8]) -> Self {
                unsafe { ($($R::from_raw(ptrs[$idx]),)+) }
            }

            #[inline(always)]
            fn shapes(self) -> Box<[&'a [usize]]> {
                Box::new([$(self.$idx.shape()),+])
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

impl<'a, R: OutputRef<'a>> OutputRefs<'a> for R {
    #[inline(always)]
    unsafe fn from_raw(ptrs: &[*mut u8]) -> Self {
        unsafe { R::from_raw(ptrs[0]) }
    }

    #[inline(always)]
    fn shapes(self) -> Box<[&'a [usize]]> {
        Box::new([self.shape()])
    }
}

impl<'a, R: OutputRef<'a>> OutputRefs<'a> for Box<[R]> {
    #[inline(always)]
    unsafe fn from_raw(ptrs: &[*mut u8]) -> Self {
        unsafe { ptrs.iter().map(|&ptr| R::from_raw(ptr)).collect() }
    }

    #[inline(always)]
    fn shapes(self) -> Box<[&'a [usize]]> {
        self.into_iter().map(|r| r.shape()).collect()
    }
}

macro_rules! impl_output_refs_for_tuple {
    ($($idx:tt: $R:ident),+ $(,)?) => {
        impl<'a, $($R: OutputRef<'a>),+> OutputRefs<'a> for ($($R,)+) {
            #[inline(always)]
            unsafe fn from_raw(ptrs: &[*mut u8]) -> Self {
                unsafe { ($($R::from_raw(ptrs[$idx]),)+) }
            }

            #[inline(always)]
            fn shapes(self) -> Box<[&'a [usize]]> {
                Box::new([$(self.$idx.shape()),+])
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
