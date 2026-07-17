use std::marker::PhantomData;

use super::{FlatRead, Interface, Value, ViewPort, ViewPorts};

/// Pokeable handle to a source node: the first element of the
/// `(node handle, port handles)` pair returned by
/// [`source`](super::Builder::source). It identifies the node whose state
/// cell [`state_mut`](super::Graph::state_mut) pokes (`S` tags the segment
/// type so the state comes back as `&mut S::State`); the pair's second
/// element carries the ordinary output [`PortHandle`]s used for wiring.
///
/// Only [`source`](super::Builder::source) mints these, so the type system
/// keeps mid-graph nodes unpokeable: a poked node reruns without any input
/// notifying, which a non-source segment's notification contract (*no-notify
/// ⟹ output unchanged*) cannot survive.
///
/// The phantom is `fn() -> S` so a handle is unconditionally `Send + Sync`
/// (it is just a node index; `S` only tags the segment type).
pub struct NodeHandle<S> {
    node: usize,
    _phantom: PhantomData<fn() -> S>,
}

impl<S> NodeHandle<S> {
    pub(crate) fn new(node: usize) -> Self {
        Self {
            node,
            _phantom: PhantomData,
        }
    }

    /// The engine node index of the source.
    pub fn index(&self) -> usize {
        self.node
    }
}

// Manual (not derived) so it never requires `S: Debug` -- a `NodeHandle` is
// just a node index, and `S` is only a phantom tag.
impl<S> std::fmt::Debug for NodeHandle<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeHandle({})", self.node)
    }
}

impl<S> Clone for NodeHandle<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for NodeHandle<S> {}

/// Typed handle to a slot in the graph.
///
/// The phantom is `fn() -> T` so a handle is unconditionally `Send + Sync`
/// (it is just an index; `T` only tags the slot's port type).
pub struct PortHandle<T> {
    index: usize,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> PortHandle<T> {
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

// Manual (not derived) so it never requires `T: Debug` -- a `PortHandle` is
// just an index, and `T` is only a phantom tag.
impl<T> std::fmt::Debug for PortHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PortHandle({})", self.index)
    }
}

impl<T> Clone for PortHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for PortHandle<T> {}

/// Maps an [`Interface`] to its handle trees — the *forward* half of the
/// [`Interface`] ⇆ handle-tree bijection. The *backward* half is the inverse
/// trait [`HandlesInterface`], and the two reference each other as inverses:
/// `Handles<'a>` here is constrained to recover `Self`, while
/// [`HandlesInterface::Interface`] recovers the interface from a handle tree.
///
/// The walk-to-flat direction (reading a handle tree into indices) lives wholly
/// on [`HandlesInterface`] — it is a property of the handles you *have*; this
/// trait owns only the build-from-flat direction (constructing the output handle
/// tree once the engine has assigned its cell indices).
pub trait InterfaceHandles: Interface {
    /// The borrowed input handle tree; its inverse [`HandlesInterface`] recovers
    /// `Self`, witnessing the bijection.
    type Handles<'a>: Copy + HandlesInterface<Interface = Self>;

    /// The owned output handle tree.
    type HandlesOwned;

    /// Rebuild the output handle tree from flat output indices, sized by `shape`.
    fn handles_from_flat(
        shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Self::HandlesOwned;
}

// -- Compound: empty tuple --------------------------------------------------

impl InterfaceHandles for () {
    type Handles<'a> = ();
    type HandlesOwned = ();

    #[inline(always)]
    fn handles_from_flat(_: &mut FlatRead<usize>, _: &mut FlatRead<usize>) {}
}

// -- Compound: tuple branches (arities 1-12) --------------------------------

macro_rules! impl_handles_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: InterfaceHandles,)+> InterfaceHandles for ($($T,)+) {
            type Handles<'a> = ($($T::Handles<'a>,)+);
            type HandlesOwned = ($($T::HandlesOwned,)+);

            #[inline]
            fn handles_from_flat(shape: &mut FlatRead<usize>, indices: &mut FlatRead<usize>) -> Self::HandlesOwned {
                ( $( <$T as InterfaceHandles>::handles_from_flat(shape, indices), )+ )
            }
        }
    };
}

impl_handles_for_tuple!(0: A);
impl_handles_for_tuple!(0: A, 1: B);
impl_handles_for_tuple!(0: A, 1: B, 2: C);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_handles_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// -- Leaf: ViewPort<V> ------------------------------------------------------

impl<V: Value> InterfaceHandles for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Handles<'a> = PortHandle<ViewPort<V>>;
    type HandlesOwned = PortHandle<ViewPort<V>>;

    #[inline(always)]
    fn handles_from_flat(
        _shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> PortHandle<ViewPort<V>> {
        PortHandle::new(*indices.pop())
    }
}

// -- Variadic leaf: ViewPorts<V> ---------------------------------------------

impl<V: Value> InterfaceHandles for ViewPorts<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Handles<'a> = &'a [PortHandle<ViewPort<V>>];
    type HandlesOwned = Box<[PortHandle<ViewPort<V>>]>;

    #[inline]
    fn handles_from_flat(
        shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Self::HandlesOwned {
        let n = *shape.pop();
        (0..n).map(|_| PortHandle::new(*indices.pop())).collect()
    }
}

// ===========================================================================
// HandlesInterface — the inverse of `InterfaceHandles`
// ===========================================================================

/// Recovers the wired [`Interface`] from a handle tree — the *backward* half of
/// the [`Interface`] ⇆ handle-tree bijection, inverse to [`InterfaceHandles`]
/// (whose `Handles<'a>` reconstitutes the very handle tree this maps back).
///
/// The forward map [`InterfaceHandles::Handles`] is a projection type inference
/// cannot run backwards: a builder method whose handle parameter is spelled
/// `<O::Inputs as InterfaceHandles>::Handles<'_>` leaves `O` (and thus any of
/// `O`'s generic parameters) unconstrained by the handle argument. This trait
/// goes the other way — a [`PortHandle`] is structural in its port type, so the
/// handle *argument* determines `Self` and `Self::Interface` normalizes forward.
/// Spelling [`Builder::segment`](super::Builder::segment)'s input as a free
/// `H: HandlesInterface` with `O: Segment<Inputs = H::Interface>` therefore lets
/// an operator's generics be **inferred from the wiring** — e.g.
/// `segment(Op::new(..), handle)` with no turbofish.
pub trait HandlesInterface {
    /// The interface these handles wire; its inverse [`InterfaceHandles::Handles`]
    /// reconstitutes `Self`.
    type Interface: Interface;

    /// Write every leaf's node index into `indices` in tree-order, pushing each
    /// variadic group's ([`ViewPorts`]) element count into
    /// `shape`. The walk-to-flat direction, owned wholly by this (handle-keyed)
    /// side.
    fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>);
}

// -- Compound: empty tuple --------------------------------------------------

impl HandlesInterface for () {
    type Interface = ();

    #[inline(always)]
    fn to_vec(self, _shape: &mut Vec<usize>, _indices: &mut Vec<usize>) {}
}

// -- Compound: tuple branches (arities 1-12) --------------------------------

macro_rules! impl_handles_interface_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: HandlesInterface,)+> HandlesInterface for ($($T,)+) {
            type Interface = ($($T::Interface,)+);

            #[inline]
            fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
                $( self.$idx.to_vec(shape, indices); )+
            }
        }
    };
}

impl_handles_interface_for_tuple!(0: A);
impl_handles_interface_for_tuple!(0: A, 1: B);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_handles_interface_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

// -- Leaf: PortHandle<ViewPort<V>> -------------------------------------------

impl<V: Value> HandlesInterface for PortHandle<ViewPort<V>>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Interface = ViewPort<V>;

    #[inline(always)]
    fn to_vec(self, _shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        indices.push(self.index);
    }
}

// -- Variadic leaf: &[PortHandle<ViewPort<V>>] --------------------------------

impl<V: Value> HandlesInterface for &[PortHandle<ViewPort<V>>]
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Interface = ViewPorts<V>;

    #[inline]
    fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(self.len());
        indices.extend(self.iter().map(|h| h.index));
    }
}

// A reference to a fixed-size array of handles, e.g. `&[a, b, c]`. With a free
// handle type parameter there is no slice-typed parameter to unsize-coerce into,
// so the array case is supported explicitly (delegating to the slice logic).
impl<V: Value, const M: usize> HandlesInterface for &[PortHandle<ViewPort<V>>; M]
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Interface = ViewPorts<V>;

    #[inline]
    fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(M);
        indices.extend(self.iter().map(|h| h.index));
    }
}
