//! Typed handles and input-handle trait machinery.
//!
//! [`Handle<T>`] is a lightweight index into a [`Scenario`](super::Scenario)'s
//! node storage, parameterised by the scalar type.  [`InputKindsHandles`] maps
//! operator input types to their corresponding handle types for
//! registration-time validation.

use std::any::TypeId;
use std::marker::PhantomData;

use crate::store::Store;
use crate::types::{InputKinds, Scalar};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A typed handle into a [`Scenario`](super::Scenario)'s node storage.
///
/// The type parameter encodes the scalar type of the node's store.
///
/// Handles carry only an index; type safety is enforced by [`TypeId`] checks
/// at registration time.
#[derive(Debug)]
pub struct Handle<T> {
    pub(super) index: usize,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

// ===========================================================================
// Store-based handle mapping
// ===========================================================================

/// Maps an [`InputKinds`] collection to its corresponding
/// [`Handle`] collection, enabling registration-time validation.
///
/// Provides `node_ids` which extracts `(node_index, TypeId)` pairs from
/// handles for TypeId validation and store-pointer collection.
pub trait InputKindsHandles: InputKinds {
    /// The handle collection (e.g. `(Handle<f64>, Handle<f64>)`).
    type Handles;

    /// Extract `(node_index, scalar_type_id)` from each handle.
    fn node_ids(handles: &Self::Handles) -> Box<[(usize, TypeId)]>;
}

// -- Single store ------------------------------------------------------------

impl<T: Scalar> InputKindsHandles for Store<T> {
    type Handles = Handle<T>;

    fn node_ids(handle: &Handle<T>) -> Box<[(usize, TypeId)]> {
        Box::new([(handle.index, TypeId::of::<T>())])
    }
}

// -- Homogeneous slice -------------------------------------------------------

impl<T: Scalar> InputKindsHandles for [Store<T>] {
    type Handles = Box<[Handle<T>]>;

    fn node_ids(handles: &Box<[Handle<T>]>) -> Box<[(usize, TypeId)]> {
        handles
            .iter()
            .map(|h| (h.index, TypeId::of::<T>()))
            .collect()
    }
}

// -- Tuples (macro-generated) ------------------------------------------------

macro_rules! impl_input_kinds_handles_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Scalar),+> InputKindsHandles for ($(Store<$T>,)+) {
            type Handles = ($(Handle<$T>,)+);

            fn node_ids(handles: &Self::Handles) -> Box<[(usize, TypeId)]> {
                Box::new([$((handles.$idx.index, TypeId::of::<$T>())),+])
            }
        }
    };
}

impl_input_kinds_handles_tuple!(0: A);
impl_input_kinds_handles_tuple!(0: A, 1: B);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_input_kinds_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
