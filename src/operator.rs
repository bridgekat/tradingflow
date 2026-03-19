use super::observable::Observable;
use super::scenario::{ObservableHandle, SeriesHandle};
use super::series::Series;

/// An operator reads from one or more inputs and writes into an output buffer.
pub trait Operator {
    /// Borrowed views of the operator's inputs at compute time.
    type Inputs<'a>: InputRefs<'a>;

    /// Scalar type written into the output array.
    type Scalar: Copy;

    /// Compute the output element shape from input shapes.
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Box<[usize]>;

    /// Write the output from the inputs and the current internal state,
    /// or return `false` if no output is produced.
    fn compute(
        &mut self,
        timestamp: i64,
        inputs: Self::Inputs<'_>,
        out: &mut [Self::Scalar],
    ) -> bool;
}

/// Typed reference <-> raw pointer conversion for a single input.
///
/// Implemented by references to [`Observable<T>`] and [`Series<T>`].
///
/// # Safety
///
/// `from_raw` must correctly reconstruct a reference from a pointer
/// produced by `as_ptr`.
pub unsafe trait InputRef<'a> {
    /// The handle type used at registration time.
    type Handle;

    /// Extract `(node_index, is_series)` from a handle.
    fn node_id(handle: &Self::Handle) -> (usize, bool);

    /// Convert this reference to a raw pointer.
    fn as_ptr(&self) -> *const u8;

    /// Reconstruct a typed reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to the correct concrete type and be valid for `'a`.
    unsafe fn from_raw(ptr: *const u8) -> Self;

    /// The element shape of the referenced observable or series.
    fn shape(&self) -> &'a [usize];
}

unsafe impl<'a, T: Copy + 'static> InputRef<'a> for &'a Observable<T> {
    type Handle = ObservableHandle<T>;

    #[inline(always)]
    fn node_id(handle: &ObservableHandle<T>) -> (usize, bool) {
        (handle.index(), false)
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        *self as *const Observable<T> as *const u8
    }

    #[inline(always)]
    unsafe fn from_raw(ptr: *const u8) -> Self {
        unsafe { &*(ptr as *const Observable<T>) }
    }

    #[inline(always)]
    fn shape(&self) -> &'a [usize] {
        Observable::shape(self)
    }
}

unsafe impl<'a, T: Copy + 'static> InputRef<'a> for &'a Series<T> {
    type Handle = SeriesHandle<T>;

    #[inline(always)]
    fn node_id(handle: &SeriesHandle<T>) -> (usize, bool) {
        (handle.index(), true)
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const u8 {
        *self as *const Series<T> as *const u8
    }

    #[inline(always)]
    unsafe fn from_raw(ptr: *const u8) -> Self {
        unsafe { &*(ptr as *const Series<T>) }
    }

    #[inline(always)]
    fn shape(&self) -> &'a [usize] {
        Series::shape(self)
    }
}

// ---------------------------------------------------------------------------
// InputRefs
// ---------------------------------------------------------------------------

/// A collection of [`InputRef`]s with an associated `Handles` type.
///
/// # Safety
///
/// `from_ptrs` must correctly reconstruct references from pointers.
pub unsafe trait InputRefs<'a> {
    /// The handle collection type used at registration time.
    type Handles;

    /// Extract `(node_index, is_series)` from each handle.
    fn node_ids(handles: &Self::Handles) -> Box<[(usize, bool)]>;

    /// Reconstruct typed references from raw pointers.
    ///
    /// # Safety
    ///
    /// `ptrs` must point to `n` valid entries of the correct types.
    unsafe fn from_raw(ptrs: *const *const u8, n: usize) -> Self;

    /// Collect input shapes.
    fn shapes(&self) -> Box<[&'a [usize]]>;
}

// -- Box<[R]> (homogeneous variable-arity) ---------------------------------

unsafe impl<'a, R: InputRef<'a>> InputRefs<'a> for Box<[R]> {
    type Handles = Box<[R::Handle]>;

    fn node_ids(handles: &Box<[R::Handle]>) -> Box<[(usize, bool)]> {
        handles.iter().map(|h| R::node_id(h)).collect()
    }

    unsafe fn from_raw(ptrs: *const *const u8, n: usize) -> Self {
        let mut refs = Vec::with_capacity(n);
        for i in 0..n {
            refs.push(unsafe { R::from_raw(*ptrs.add(i)) });
        }
        refs.into()
    }

    fn shapes(&self) -> Box<[&'a [usize]]> {
        self.iter().map(|r| r.shape()).collect()
    }
}

// -- Tuples (heterogeneous fixed-arity) ------------------------------------

macro_rules! impl_input_refs_tuple {
    ($($idx:tt: $R:ident),+ $(,)?) => {
        unsafe impl<'a, $($R: InputRef<'a>),+> InputRefs<'a> for ($($R,)+) {
            type Handles = ($($R::Handle,)+);

            fn node_ids(handles: &Self::Handles) -> Box<[(usize, bool)]> {
                Box::new([$($R::node_id(&handles.$idx)),+])
            }

            unsafe fn from_raw(ptrs: *const *const u8, _n: usize) -> Self {
                unsafe { ($($R::from_raw(*ptrs.add($idx)),)+) }
            }

            fn shapes(&self) -> Box<[&'a [usize]]> {
                Box::new([$(self.$idx.shape()),+])
            }
        }
    };
}

impl_input_refs_tuple!(0: A);
impl_input_refs_tuple!(0: A, 1: B);
impl_input_refs_tuple!(0: A, 1: B, 2: C);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O);
impl_input_refs_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L, 12: M, 13: N, 14: O, 15: P);
