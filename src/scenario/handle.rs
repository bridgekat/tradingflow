//! Typed handles and input-handle trait machinery.
//!
//! [`Handle<T>`] is a lightweight index into a [`Scenario`](super::Scenario)'s
//! node storage, parameterised by the value type.  [`InputTypesHandles`] maps
//! operator input types to their corresponding handle types.

use std::marker::PhantomData;

use crate::types::InputTypes;

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A typed handle into a [`Scenario`](super::Scenario)'s node storage.
///
/// The type parameter encodes the value type of the node.
///
/// Handles carry only an index; type safety is enforced by [`TypeId`] checks
/// at registration time via [`InputTypes::type_ids`].
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
// Handle mapping
// ===========================================================================

/// Maps an [`InputTypes`] collection to its corresponding [`Handle`]
/// collection.
///
/// `node_indices` extracts the raw node index from each handle.
/// TypeId validation is handled separately by [`InputTypes::type_ids`].
pub trait InputTypesHandles: InputTypes {
    /// The handle collection (e.g. `(Handle<Array<f64>>, Handle<Array<f64>>)`).
    type Handles;

    /// Extract node indices from handles.
    fn node_indices(handles: &Self::Handles) -> Box<[usize]>;
}

// -- Single input ------------------------------------------------------------

impl<A: Send + 'static> InputTypesHandles for (A,) {
    type Handles = (Handle<A>,);

    fn node_indices(handles: &(Handle<A>,)) -> Box<[usize]> {
        Box::new([handles.0.index])
    }
}

// -- Homogeneous slice -------------------------------------------------------

impl<T: Send + 'static> InputTypesHandles for [T] {
    type Handles = Box<[Handle<T>]>;

    fn node_indices(handles: &Box<[Handle<T>]>) -> Box<[usize]> {
        handles.iter().map(|h| h.index).collect()
    }
}

// -- Tuples (macro-generated) ------------------------------------------------

macro_rules! impl_input_kinds_handles_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: Send + 'static),+> InputTypesHandles for ($($T,)+) {
            type Handles = ($(Handle<$T>,)+);

            fn node_indices(handles: &Self::Handles) -> Box<[usize]> {
                Box::new([$(handles.$idx.index),+])
            }
        }
    };
}

// Arity 1 is already covered above.
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
