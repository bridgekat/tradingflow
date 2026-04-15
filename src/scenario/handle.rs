//! Typed handles and input-handle trait machinery.
//!
//! [`Handle<T>`] is a lightweight index into a [`Scenario`](super::Scenario)'s
//! node storage, parameterised by the value type.  [`InputTypesHandles`] maps
//! the nested [`InputTypes`] tree of an operator's `Inputs` declaration to
//! the nested tree of user-supplied handles, and provides the machinery to
//! flatten both (node indices and the runtime shape) for scenario
//! registration.

use std::marker::PhantomData;

use crate::data::{FlatWrite, Input, InputTypes, Slice, SliceShape};

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// A typed handle into a [`Scenario`](super::Scenario)'s node storage.
///
/// The type parameter encodes the value type of the node.
///
/// Handles carry only an index; type safety is enforced by [`TypeId`]
/// checks at registration time via
/// [`InputTypes::write_type_ids`](crate::data::InputTypes::write_type_ids).
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
    /// Create a handle from a raw node index.
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// Returns the underlying node index.
    pub fn index(&self) -> usize {
        self.index
    }
}

// ===========================================================================
// Handle mapping
// ===========================================================================

/// Maps an [`InputTypes`] tree to its corresponding handle tree.
///
/// Parallels [`InputTypes`] exactly: leaves map to [`Handle<T>`]; tuples map
/// to tuples of child handles; slices map to `Vec<H>` of child handles.
/// Provides three methods that flatten the handle tree into the forms
/// needed for scenario registration:
///
/// * [`arity`](Self::arity) — total flat leaf count.
/// * [`write_node_indices`](Self::write_node_indices) — write the node
///   index of every leaf into a caller-allocated buffer (tree order).
/// * [`shape`](Self::shape) — build the runtime [`InputTypes::Shape`] from
///   the handle tree (e.g. slice lengths flow from handle vecs into
///   [`SliceShape`]).
pub trait InputTypesHandles: InputTypes {
    /// The handle tree (e.g. `(Handle<A>, Handle<B>)` for
    /// `(Input<A>, Input<B>)`).
    type Handles;

    /// Total number of flat leaf positions.
    fn arity(handles: &Self::Handles) -> usize;

    /// Write the node index of each leaf into `writer` in tree-order.
    ///
    /// Advances the cursor by exactly [`arity(handles)`](Self::arity)
    /// slots.  Cursor-style traversal avoids recomputing sub-tree
    /// arities — total cost is O(total_leaves).
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>);

    /// Build the runtime [`InputTypes::Shape`] from the handle tree.
    fn shape(handles: &Self::Handles) -> Self::Shape;
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

    #[inline(always)]
    fn shape(_: &Handle<T>) {}
}

// -- Compound: empty tuple (arity 0) ----------------------------------------

impl InputTypesHandles for () {
    type Handles = ();

    #[inline(always)]
    fn arity(_: &()) -> usize {
        0
    }

    #[inline(always)]
    fn write_node_indices(_: &(), _writer: &mut FlatWrite<usize>) {}

    #[inline(always)]
    fn shape(_: &()) {}
}

// -- Compound: tuple branches (arities 1-12) --------------------------------

macro_rules! impl_input_types_handles_for_tuple {
    ($($idx:tt: $T:ident),+ $(,)?) => {
        impl<$($T: InputTypesHandles),+> InputTypesHandles for ($($T,)+) {
            type Handles = ($($T::Handles,)+);

            #[inline]
            fn arity(handles: &Self::Handles) -> usize {
                0 $(+ <$T as InputTypesHandles>::arity(&handles.$idx))+
            }

            #[inline]
            fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
                $(
                    <$T as InputTypesHandles>::write_node_indices(&handles.$idx, writer);
                )+
            }

            #[inline]
            fn shape(handles: &Self::Handles) -> Self::Shape {
                ($(<$T as InputTypesHandles>::shape(&handles.$idx),)+)
            }
        }
    };
}

impl_input_types_handles_for_tuple!(0: A);
impl_input_types_handles_for_tuple!(0: A, 1: B);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_input_types_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_input_types_handles_for_tuple!(
    0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K
);
impl_input_types_handles_for_tuple!(
    0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L
);

// -- Compound: slice branch (Slice<T>) --------------------------------------

impl<T: InputTypesHandles + 'static> InputTypesHandles for Slice<T> {
    type Handles = Vec<T::Handles>;

    #[inline]
    fn arity(handles: &Self::Handles) -> usize {
        handles
            .iter()
            .map(|h| <T as InputTypesHandles>::arity(h))
            .sum()
    }

    #[inline]
    fn write_node_indices(handles: &Self::Handles, writer: &mut FlatWrite<usize>) {
        for h in handles.iter() {
            <T as InputTypesHandles>::write_node_indices(h, writer);
        }
    }

    #[inline]
    fn shape(handles: &Self::Handles) -> Self::Shape {
        let elems: Vec<T::Shape> = handles
            .iter()
            .map(|h| <T as InputTypesHandles>::shape(h))
            .collect();
        SliceShape::new(elems.into_boxed_slice())
    }
}
