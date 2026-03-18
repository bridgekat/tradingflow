//! Input slot types and tuple machinery for heterogeneous operator inputs.
//!
//! Each operator input position is independently typed — both in element type
//! `T` and in access mode:
//!
//! * [`Obs<T>`] — read the latest value from an [`Observable<T>`].
//! * [`Hist<T>`] — read the full history from a [`Series<T>`].
//!
//! The [`InputSlot`] trait maps each marker to its handle type and reference
//! type for single-element access (used by [`InputTuple`]).
//!
//! The [`InputSlice`] trait provides bulk reconstruction of a raw pointer
//! array into a typed slice (used by [`Scenario::add_slice_operator`]).

use std::marker::PhantomData;

use crate::observable::{Observable, ObservableHandle};
use crate::series::{Series, SeriesHandle};

// ---------------------------------------------------------------------------
// Marker types
// ---------------------------------------------------------------------------

/// Marker: operator reads the latest value from an observable.
pub struct Obs<T: Copy>(PhantomData<T>);

/// Marker: operator reads full history from a materialized series.
pub struct Hist<T: Copy>(PhantomData<T>);

// ---------------------------------------------------------------------------
// InputSlot — per-element access (for InputTuple)
// ---------------------------------------------------------------------------

/// Maps a marker type to its handle type and reference type.
///
/// Implemented for [`Obs<T>`] and [`Hist<T>`].
pub trait InputSlot {
    /// The handle type the user provides at registration.
    type Handle: Copy;
    /// The reference type the operator receives at compute time.
    type Ref<'a>
    where
        Self: 'a;

    /// Extract the raw pointer from the scenario's node storage.
    ///
    /// # Safety contract (logical)
    ///
    /// The returned pointer must remain valid for the lifetime of the scenario.
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], handle: Self::Handle) -> *mut u8;

    /// Reconstruct a reference from a raw pointer.
    ///
    /// # Safety
    ///
    /// `ptr` must point to the correct concrete type.
    unsafe fn from_ptr<'a>(ptr: *mut u8) -> Self::Ref<'a>;

    /// The node index referenced by this handle.
    fn node_index(handle: Self::Handle) -> usize;
}

impl<T: Copy> InputSlot for Obs<T> {
    type Handle = ObservableHandle<T>;
    type Ref<'a>
        = &'a Observable<T>
    where
        T: 'a;

    #[inline]
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], h: ObservableHandle<T>) -> *mut u8 {
        nodes[h.index].obs_ptr
    }

    #[inline]
    unsafe fn from_ptr<'a>(ptr: *mut u8) -> &'a Observable<T> {
        unsafe { &*(ptr as *const Observable<T>) }
    }

    #[inline]
    fn node_index(h: ObservableHandle<T>) -> usize {
        h.index
    }
}

impl<T: Copy> InputSlot for Hist<T> {
    type Handle = SeriesHandle<T>;
    type Ref<'a>
        = &'a Series<T>
    where
        T: 'a;

    #[inline]
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], h: SeriesHandle<T>) -> *mut u8 {
        // SeriesHandle<T> is only constructible via Scenario::materialize(),
        // which sets series_ptr to non-null.
        nodes[h.index].series_ptr
    }

    #[inline]
    unsafe fn from_ptr<'a>(ptr: *mut u8) -> &'a Series<T> {
        unsafe { &*(ptr as *const Series<T>) }
    }

    #[inline]
    fn node_index(h: SeriesHandle<T>) -> usize {
        h.index
    }
}

// ---------------------------------------------------------------------------
// InputSlice — bulk slice access (for add_slice_operator)
// ---------------------------------------------------------------------------

/// Maps a marker type to its handle type and slice-of-references type.
///
/// Like [`InputSlot`] but for variable-arity homogeneous inputs.
/// Provides `slice_from_ptrs` for zero-copy reconstruction of a raw
/// pointer array into `&[Ref<'_>]`.
pub trait InputSlice {
    /// The handle type the user provides at registration.
    type Handle: Copy;
    /// The element reference type (the slice element).
    type Ref<'a>
    where
        Self: 'a;

    /// Extract the raw pointer from the scenario's node storage.
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], handle: Self::Handle) -> *mut u8;

    /// Reinterpret a raw pointer array as a slice of references.
    ///
    /// This exploits the fact that `Ref<'a>` (a shared reference) has the
    /// same size and alignment as `*mut u8`.
    ///
    /// # Safety
    ///
    /// * Each pointer in `ptrs[0..n]` must point to the correct concrete type.
    /// * `Ref<'a>` must be pointer-sized (true for all reference types).
    unsafe fn slice_from_ptrs<'a>(ptrs: *const *mut u8, n: usize) -> &'a [Self::Ref<'a>];

    /// The node index referenced by this handle.
    fn node_index(handle: Self::Handle) -> usize;
}

impl<T: Copy> InputSlice for Obs<T> {
    type Handle = ObservableHandle<T>;
    type Ref<'a>
        = &'a Observable<T>
    where
        T: 'a;

    #[inline]
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], h: ObservableHandle<T>) -> *mut u8 {
        nodes[h.index].obs_ptr
    }

    #[inline]
    unsafe fn slice_from_ptrs<'a>(ptrs: *const *mut u8, n: usize) -> &'a [&'a Observable<T>] {
        unsafe { std::slice::from_raw_parts(ptrs as *const &Observable<T>, n) }
    }

    #[inline]
    fn node_index(h: ObservableHandle<T>) -> usize {
        h.index
    }
}

impl<T: Copy> InputSlice for Hist<T> {
    type Handle = SeriesHandle<T>;
    type Ref<'a>
        = &'a Series<T>
    where
        T: 'a;

    #[inline]
    fn extract_ptr(nodes: &[crate::scenario::NodeSlot], h: SeriesHandle<T>) -> *mut u8 {
        nodes[h.index].series_ptr
    }

    #[inline]
    unsafe fn slice_from_ptrs<'a>(ptrs: *const *mut u8, n: usize) -> &'a [&'a Series<T>] {
        unsafe { std::slice::from_raw_parts(ptrs as *const &Series<T>, n) }
    }

    #[inline]
    fn node_index(h: SeriesHandle<T>) -> usize {
        h.index
    }
}

// ---------------------------------------------------------------------------
// InputTuple
// ---------------------------------------------------------------------------

/// Converts a tuple of input slots to/from raw pointers for type erasure.
pub trait InputTuple {
    /// Tuple of handle types (one per input position).
    type Handles: Copy;
    /// Tuple of reference types (one per input position).
    type Refs<'a>
    where
        Self: 'a;
    /// Number of input positions.
    const N: usize;

    /// Collect raw pointers from handles via the scenario's node storage.
    fn extract_ptrs(nodes: &[crate::scenario::NodeSlot], handles: Self::Handles) -> Vec<*mut u8>;

    /// Reconstruct the reference tuple from raw pointers.
    ///
    /// # Safety
    ///
    /// Each pointer in `ptrs[0..N]` must point to the correct concrete type.
    unsafe fn from_ptrs<'a>(ptrs: *const *mut u8) -> Self::Refs<'a>;

    /// Collect node indices from handles.
    fn node_indices(handles: Self::Handles) -> Vec<usize>;
}

// ---------------------------------------------------------------------------
// Macro-generated tuple impls
// ---------------------------------------------------------------------------

macro_rules! impl_input_tuple {
    ($($idx:tt: $slot:ident),+ $(,)?) => {
        impl<$($slot: InputSlot),+> InputTuple for ($($slot,)+) {
            type Handles = ($($slot::Handle,)+);
            type Refs<'a> = ($($slot::Ref<'a>,)+) where $($slot: 'a),+;
            const N: usize = impl_input_tuple!(@count $($slot),+);

            fn extract_ptrs(
                nodes: &[crate::scenario::NodeSlot],
                handles: Self::Handles,
            ) -> Vec<*mut u8> {
                vec![$($slot::extract_ptr(nodes, handles.$idx)),+]
            }

            unsafe fn from_ptrs<'a>(ptrs: *const *mut u8) -> Self::Refs<'a> {
                unsafe { ($($slot::from_ptr::<'a>(*ptrs.add($idx)),)+) }
            }

            fn node_indices(handles: Self::Handles) -> Vec<usize> {
                vec![$($slot::node_index(handles.$idx)),+]
            }
        }
    };
    // Counting helper
    (@count $first:ident $(, $rest:ident)*) => {
        1usize $(+ impl_input_tuple!(@one $rest))*
    };
    (@one $_:ident) => { 1usize };
}

impl_input_tuple!(0: A);
impl_input_tuple!(0: A, 1: B);
impl_input_tuple!(0: A, 1: B, 2: C);
impl_input_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
