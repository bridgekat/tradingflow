use std::any::TypeId;

// ===========================================================================
// Generalized input system
// ===========================================================================

/// A collection of typed inputs for operators.
///
/// Provides:
///
/// * [`Refs<'a>`](InputTypes::Refs) — aggregated immutable references to input
///   values (e.g. `(&'a Array<f64>, &'a Array<f64>)` for a binary operator).
/// * [`type_ids`](InputTypes::type_ids) — expected `TypeId`s for each input
///   position, enabling runtime validation without typed handles.
///
/// Implemented for single values `(T,)`, tuples of values up to arity 12,
/// and homogeneous slices `[T]`.
pub trait InputTypes {
    /// Aggregated immutable references.
    type Refs<'a>;

    /// Reconstruct typed references from raw value pointers.
    ///
    /// # Safety
    ///
    /// Each `ptrs[i]` must point to a valid value whose type matches the
    /// corresponding position in `Self`.
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a>;

    /// Expected `TypeId` for each input position.
    ///
    /// For tuples, returns one `TypeId` per position (ignores `arity`).
    /// For homogeneous slices `[T]`, returns `arity` copies of
    /// `TypeId::of::<T>()`.
    fn type_ids(arity: usize) -> Box<[TypeId]>;
}

// -- Homogeneous slice input -------------------------------------------------

impl<T: Send + 'static> InputTypes for [T] {
    type Refs<'a> = Box<[&'a T]>;

    #[inline(always)]
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Box<[&'a T]> {
        unsafe { ptrs.iter().map(|&p| &*(p as *const T)).collect() }
    }

    #[inline(always)]
    fn type_ids(arity: usize) -> Box<[TypeId]> {
        vec![TypeId::of::<T>(); arity].into()
    }
}

// -- Empty input (0-arity) ---------------------------------------------------

impl InputTypes for () {
    type Refs<'a> = ();

    #[inline(always)]
    unsafe fn from_ptrs<'a>(_ptrs: &[*const u8]) -> Self::Refs<'a> {}

    #[inline(always)]
    fn type_ids(_arity: usize) -> Box<[TypeId]> {
        Box::new([])
    }
}

// -- Tuple inputs (macro-generated) ------------------------------------------

macro_rules! impl_input_types_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Send + 'static),+> InputTypes for ($($T,)+) {
            type Refs<'a> = ($(&'a $T,)+);

            #[inline(always)]
            unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a> {
                unsafe { ($(&*(ptrs[$idx] as *const $T),)+) }
            }

            #[inline(always)]
            fn type_ids(_arity: usize) -> Box<[TypeId]> {
                Box::new([$(TypeId::of::<$T>(),)+])
            }
        }
    };
}

impl_input_types_for_tuple!(0: A);
impl_input_types_for_tuple!(0: A, 1: B);
impl_input_types_for_tuple!(0: A, 1: B, 2: C);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_types_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
