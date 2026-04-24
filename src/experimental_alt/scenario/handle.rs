//! Typed handles and input-handle trait machinery.
//!
//! [`Handle<T>`] is a lightweight index into a [`Scenario`](super::Scenario)'s
//! node storage, parameterised by the value type.  [`InputTypesHandles`]
//! maps the nested [`InputTypes`](crate::data::InputTypes) tree of an
//! operator's `Inputs` declaration to the corresponding tree of
//! user-supplied handles, and flattens both node indices and type-id
//! bookkeeping for scenario registration.
//!
//! This is a verbatim port of `src/scenario/handle.rs` with the only
//! change being the re-export path for `InputTypes`.

use std::marker::PhantomData;

use crate::data::{FlatWrite, Input, InputTypes};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A typed handle into a [`Scenario`](super::Scenario)'s node storage.
#[derive(Debug)]
pub struct Handle<T> {
    pub(crate) index: usize,
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

// -- Leaf: Input<T> ---------------------------------------------------------

impl<T: Send + 'static> InputTypesHandles for Input<T> {
    type Handles = Handle<T>;

    #[inline]
    fn arity(_: &Self::Handles) -> usize {
        1
    }

    #[inline]
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
        writer.push(handles.index);
    }
}

// -- () — zero inputs -------------------------------------------------------

impl InputTypesHandles for () {
    type Handles = ();

    #[inline]
    fn arity(_: &Self::Handles) -> usize {
        0
    }

    #[inline]
    fn write_node_indices(_: &Self::Handles, _: &mut FlatWrite<usize>) {}
}

// -- 1-tuple with ?Sized last element --------------------------------------

impl<S: InputTypesHandles + ?Sized> InputTypesHandles for (S,) {
    type Handles = (S::Handles,);

    #[inline]
    fn arity(handles: &Self::Handles) -> usize {
        <S as InputTypesHandles>::arity(&handles.0)
    }

    #[inline]
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
        S::write_node_indices(&handles.0, writer);
    }
}

// -- N-tuple with Sized prefix + possibly ?Sized last ----------------------

macro_rules! impl_handles_tuple {
    ($($idx:tt: $T:ident),+; $last_idx:tt: $S:ident) => {
        impl<$($T: InputTypesHandles,)+ $S: InputTypesHandles + ?Sized>
            InputTypesHandles for ($($T,)+ $S,)
        {
            type Handles = ($($T::Handles,)+ $S::Handles,);

            #[inline]
            fn arity(handles: &Self::Handles) -> usize {
                0
                $(+ <$T as InputTypesHandles>::arity(&handles.$idx))+
                + <$S as InputTypesHandles>::arity(&handles.$last_idx)
            }

            #[inline]
            fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
                $( $T::write_node_indices(&handles.$idx, writer); )+
                $S::write_node_indices(&handles.$last_idx, writer);
            }
        }
    };
}

impl_handles_tuple!(0: A; 1: B);
impl_handles_tuple!(0: A, 1: B; 2: C);
impl_handles_tuple!(0: A, 1: B, 2: C; 3: D);
impl_handles_tuple!(0: A, 1: B, 2: C, 3: D; 4: E);
impl_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E; 5: F);
impl_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F; 6: G);
impl_handles_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G; 7: H);

// -- [T] trailing slice (variadic) -----------------------------------------

impl<T: InputTypesHandles + 'static> InputTypesHandles for [T]
where
    T::Handles: Sized,
{
    type Handles = Vec<T::Handles>;

    #[inline]
    fn arity(handles: &Self::Handles) -> usize {
        handles.iter().map(|h| <T as InputTypesHandles>::arity(h)).sum()
    }

    #[inline]
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
        for h in handles {
            T::write_node_indices(h, writer);
        }
    }
}
