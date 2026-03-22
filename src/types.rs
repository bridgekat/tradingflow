/// A permitted array scalar type.
pub trait Scalar: Sized + Send + Sync + Clone + Default + 'static {}

macro_rules! impl_scalar {
    ($($T:ty),+ $(,)?) => { $(impl Scalar for $T {})+ };
}

impl_scalar!((), bool, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

// ===========================================================================
// Generalized input system
// ===========================================================================

/// A collection of typed inputs for operators.
///
/// Provides:
///
/// * `Refs<'a>` — aggregated immutable references to input values
///   (e.g. `(&'a ArrayD<f64>, &'a ArrayD<f64>)` for a binary operator).
///
/// Implemented for single values `(T,)`, tuples of values up to arity 12,
/// and homogeneous slices `[T]`.
pub trait InputKinds {
    /// Aggregated immutable references.
    type Refs<'a>;

    /// Reconstruct typed references from raw value pointers.
    ///
    /// # Safety
    ///
    /// Each `ptrs[i]` must point to a valid value whose type matches the
    /// corresponding position in `Self`.
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a>;
}

// -- Single input ------------------------------------------------------------

impl<A: Send + 'static> InputKinds for (A,) {
    type Refs<'a> = (&'a A,);

    #[inline(always)]
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> (&'a A,) {
        unsafe { (&*(ptrs[0] as *const A),) }
    }
}

// -- Homogeneous slice input -------------------------------------------------

impl<T: Send + 'static> InputKinds for [T] {
    type Refs<'a> = Box<[&'a T]>;

    #[inline(always)]
    unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Box<[&'a T]> {
        unsafe { ptrs.iter().map(|&p| &*(p as *const T)).collect() }
    }
}

// -- Tuple inputs (macro-generated) ------------------------------------------

macro_rules! impl_input_kinds_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Send + 'static),+> InputKinds for ($($T,)+) {
            type Refs<'a> = ($(&'a $T,)+);

            #[inline(always)]
            unsafe fn from_ptrs<'a>(ptrs: &[*const u8]) -> Self::Refs<'a> {
                unsafe { ($(&*(ptrs[$idx] as *const $T),)+) }
            }
        }
    };
}

// Arity 1 is already covered by the single-input impl above.
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
