//! Typed handles and input-handle trait machinery.
//!
//! [`Handle<T>`] is a lightweight index into a [`Scenario`](super::Scenario)'s
//! node storage, parameterised by the value type.  [`InputTypesHandles`] maps
//! the nested [`InputTypes`] tree of an operator's `Inputs` declaration to
//! the corresponding tree of user-supplied handles, and flattens both node
//! indices and type-id bookkeeping for scenario registration.

use std::marker::PhantomData;

use crate::{FlatWrite, Input, InputTypes};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A typed handle into a [`Scenario`](super::Scenario)'s node storage.
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
// InputTypesHandles
// ===========================================================================

/// Maps an [`InputTypes`] tree to its corresponding handle tree.
pub trait InputTypesHandles: InputTypes {
    /// The handle tree.
    type Handles;

    /// Total flat leaf count from the handles.
    fn arity(handles: &Self::Handles) -> usize;

    /// Write every leaf's node index into `writer` in tree-order.
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>);
}

// -- Leaf: Input<T> ----------------------------------------------------------

impl<T: Send + 'static> InputTypesHandles for Input<T> {
    type Handles = Handle<T>;

    #[inline(always)]
    fn arity(_: &Handle<T>) -> usize {
        1
    }

    #[inline(always)]
    fn write_node_indices(handles: &Handle<T>, writer: &mut FlatWrite<usize>) {
        writer.push(handles.index);
    }
}

// -- Compound: empty tuple ---------------------------------------------------

impl InputTypesHandles for () {
    type Handles = ();

    #[inline(always)]
    fn arity(_: &()) -> usize {
        0
    }

    #[inline(always)]
    fn write_node_indices(_: &(), _writer: &mut FlatWrite<usize>) {}
}

// -- Compound: tuple branches (arities 1-12) ---------------------------------
//
// Single macro, no element distinctions.  `$T::Handles` resolves to the
// right type automatically: `Handle<T>` for `Input<T>` (Sized leaf) and
// `Vec<T::Handles>` for `[T]` (trailing slice).  The same `arity` and
// `write_node_indices` logic handles both uniformly.

// Mirrors `impl_input_types_for_tuple` exactly: prefix `Sized`, last `?Sized`.

macro_rules! impl_input_types_handles_for_tuple {
    // 1-tuple.
    ($last_idx:tt: $Last:ident) => {
        impl<$Last: InputTypesHandles + ?Sized> InputTypesHandles for ($Last,) {
            type Handles = ($Last::Handles,);

            #[inline]
            fn arity(handles: &Self::Handles) -> usize {
                <$Last as InputTypesHandles>::arity(&handles.0)
            }

            #[inline]
            fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
                <$Last as InputTypesHandles>::write_node_indices(&handles.0, writer);
            }
        }
    };
    // N-tuple: Sized prefix + ?Sized last.
    ($($idx:tt: $T:ident),+; $last_idx:tt: $Last:ident) => {
        impl<$($T: InputTypesHandles,)+ $Last: InputTypesHandles + ?Sized>
            InputTypesHandles for ($($T,)+ $Last,)
        {
            type Handles = ($($T::Handles,)+ $Last::Handles,);

            #[inline]
            fn arity(handles: &Self::Handles) -> usize {
                0 $( + <$T as InputTypesHandles>::arity(&handles.$idx) )+
                    + <$Last as InputTypesHandles>::arity(&handles.$last_idx)
            }

            #[inline]
            fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
                $( <$T as InputTypesHandles>::write_node_indices(&handles.$idx, writer); )+
                <$Last as InputTypesHandles>::write_node_indices(&handles.$last_idx, writer);
            }
        }
    };
}

impl_input_types_handles_for_tuple!(0: A);
impl_input_types_handles_for_tuple!(0: A; 1: B);
impl_input_types_handles_for_tuple!(0: A, 1: B; 2: C);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C; 3: D);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D; 4: E);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E; 5: F);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F; 6: G);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G; 7: H);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H; 8: I);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I; 9: J);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J; 10: K);
impl_input_types_handles_for_tuple!(
    0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K; 11: L
);

// -- Compound: trailing-only slice ([T]) ------------------------------------

impl<T: InputTypesHandles + 'static> InputTypesHandles for [T] {
    type Handles = Vec<T::Handles>;

    #[inline]
    fn arity(handles: &Self::Handles) -> usize {
        handles.len() * <T as InputTypes>::arity()
    }

    #[inline]
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
        for h in handles.iter() {
            <T as InputTypesHandles>::write_node_indices(h, writer);
        }
    }
}
