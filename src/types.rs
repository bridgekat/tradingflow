use std::any::TypeId;

use super::store::Store;

/// A permitted array scalar type.
pub trait Scalar: Copy + Send + Default + 'static {}

macro_rules! impl_scalar {
    ($($T:ty),+ $(,)?) => { $(impl Scalar for $T {})+ };
}

impl_scalar!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

// ===========================================================================
// Store-based input system
// ===========================================================================

/// A collection of [`Store`]-typed inputs for operators.
///
/// Provides three associated items:
///
/// * `Refs<'a>` — aggregated immutable references to input stores
///   (e.g. `(&Store<A>, &Store<B>)` for a binary operator).
/// * `WindowSizes` — aggregated minimum window sizes, one per input
///   (e.g. `(usize, usize)`).  `0` means "don't care"; `N > 0` means the
///   input store must retain at least `N` elements.
/// * `TypeIds` — aggregated type ids, one per input.
///
/// Implemented for [`Store<T>`], `[Store<T>]`, and tuples of stores up to
/// arity 12.
pub trait InputKinds {
    /// Aggregated immutable references (e.g. `(&'a Store<A>, &'a Store<B>)`).
    type Refs<'a>;

    /// Aggregated minimum window sizes (e.g. `(usize, usize)`).
    type WindowSizes;

    /// Aggregated type ids (e.g. `[TypeId; 2]`).
    type TypeIds: AsRef<[TypeId]>;

    /// The type ids of the scalar types.
    fn type_ids() -> Self::TypeIds;

    /// Reconstruct typed references from raw store pointers.
    ///
    /// # Safety
    ///
    /// Each `ptrs[i]` must point to a valid `Store<T>` whose scalar type
    /// matches the corresponding position in `Self`.
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a>;

    /// Promote input stores where the minimum window size is non-zero.
    ///
    /// # Safety
    ///
    /// Each `ptrs[i]` must point to a valid `Store<T>` whose scalar type
    /// matches the corresponding position in `Self`.
    unsafe fn promote(ptrs: &[*mut u8], sizes: &Self::WindowSizes);
}

// -- Single store input ------------------------------------------------------

impl<T: Scalar> InputKinds for Store<T> {
    type Refs<'a> = &'a Store<T>;
    type WindowSizes = usize;
    type TypeIds = [TypeId; 1];

    #[inline(always)]
    fn type_ids() -> [TypeId; 1] {
        [TypeId::of::<T>()]
    }

    #[inline(always)]
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> &'a Store<T> {
        unsafe { &*(ptrs[0] as *const Store<T>) }
    }

    unsafe fn promote(ptrs: &[*mut u8], &size: &usize) {
        if size > 0 {
            unsafe { (&mut *(ptrs[0] as *mut Store<T>)).ensure_min_window(size) };
        }
    }
}

// -- Homogeneous slice input -------------------------------------------------

impl<T: Scalar> InputKinds for [Store<T>] {
    type Refs<'a> = Box<[&'a Store<T>]>;
    type WindowSizes = Box<[usize]>;
    type TypeIds = Box<[TypeId]>;

    fn type_ids() -> Box<[TypeId]> {
        // Slice length is not known statically; callers use node_ids instead.
        Box::new([])
    }

    #[inline(always)]
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Box<[&'a Store<T>]> {
        unsafe { ptrs.iter().map(|&p| &*(p as *const Store<T>)).collect() }
    }

    unsafe fn promote(ptrs: &[*mut u8], sizes: &Box<[usize]>) {
        for (i, &size) in sizes.iter().enumerate() {
            if size > 0 {
                unsafe { (&mut *(ptrs[i] as *mut Store<T>)).ensure_min_window(size) };
            }
        }
    }
}

// -- Tuple inputs (macro-generated) ------------------------------------------

macro_rules! impl_input_kinds_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Scalar),+> InputKinds for ($(Store<$T>,)+) {
            type Refs<'a> = ($(&'a Store<$T>,)+);
            type WindowSizes = ($(impl_input_kinds_for_tuple!(@usize $T),)+);
            type TypeIds = [TypeId; impl_input_kinds_for_tuple!(@count $($T)+)];

            #[inline(always)]
            fn type_ids() -> Self::TypeIds {
                [$(TypeId::of::<$T>()),+]
            }

            #[inline(always)]
            unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a> {
                unsafe { ($(&*(ptrs[$idx] as *const Store<$T>),)+) }
            }

            unsafe fn promote(ptrs: &[*mut u8], sizes: &Self::WindowSizes) {
                unsafe {
                    $(if sizes.$idx > 0 {
                        (&mut *(ptrs[$idx] as *mut Store<$T>)).ensure_min_window(sizes.$idx);
                    })+
                }
            }
        }
    };
    (@usize $T:ident) => { usize };
    (@count $head:ident $($tail:ident)*) => {
        1 + impl_input_kinds_for_tuple!(@count $($tail)*)
    };
    (@count) => { 0 };
}

impl_input_kinds_for_tuple!(0: A);
impl_input_kinds_for_tuple!(0: A, 1: B);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_kinds_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
