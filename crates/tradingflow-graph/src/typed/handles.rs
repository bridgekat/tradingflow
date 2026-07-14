use std::marker::PhantomData;

use super::{FlatRead, Interface, RefViewPort, RefViewPorts, Segment, ValueView, ViewPort};

/// Typed handle to a slot in the graph.
pub struct Handle<T> {
    index: usize,
    _phantom: PhantomData<T>,
}

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

// Manual (not derived) so it never requires `T: Debug` -- a `Handle` is just an
// index, and `T` is only a phantom tag.
impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle({})", self.index)
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

/// The output handle of a single-output segment (a [`Handle`]); lets a
/// [`SourceHandle`] recover its slot index for `state_mut`.
pub trait OutputHandle: Copy {
    fn slot(&self) -> usize;
}

impl<M> OutputHandle for Handle<M> {
    fn slot(&self) -> usize {
        self.index
    }
}

/// Pokeable handle to a source node, produced by
/// [`push_source`](super::Builder::push_source). Carries the source
/// segment type `S` (so `state_mut` can reach the state) and derefs to `S`'s
/// output handle for wiring -- `Handle<RefViewPort<Scalar<T>>>` for a
/// [`RefSource<T>`](super::RefSource), `Handle<ViewPort<Scalar<T>>>` for a
/// [`Source<T>`](super::Source).
pub struct SourceHandle<S: Segment>
where
    S::Outputs: InterfaceHandles,
    <S::Outputs as InterfaceHandles>::HandlesOwned: OutputHandle,
{
    output: <S::Outputs as InterfaceHandles>::HandlesOwned,
}

impl<S: Segment> SourceHandle<S>
where
    S::Outputs: InterfaceHandles,
    <S::Outputs as InterfaceHandles>::HandlesOwned: OutputHandle,
{
    pub(crate) fn new(output: <S::Outputs as InterfaceHandles>::HandlesOwned) -> Self {
        Self { output }
    }

    pub fn index(&self) -> usize {
        self.output.slot()
    }
}

impl<S: Segment> Clone for SourceHandle<S>
where
    S::Outputs: InterfaceHandles,
    <S::Outputs as InterfaceHandles>::HandlesOwned: OutputHandle,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Segment> Copy for SourceHandle<S>
where
    S::Outputs: InterfaceHandles,
    <S::Outputs as InterfaceHandles>::HandlesOwned: OutputHandle,
{
}

impl<S: Segment> std::ops::Deref for SourceHandle<S>
where
    S::Outputs: InterfaceHandles,
    <S::Outputs as InterfaceHandles>::HandlesOwned: OutputHandle,
{
    type Target = <S::Outputs as InterfaceHandles>::HandlesOwned;

    fn deref(&self) -> &Self::Target {
        &self.output
    }
}

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

impl<V: ValueView> InterfaceHandles for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Handles<'a> = Handle<ViewPort<V>>;
    type HandlesOwned = Handle<ViewPort<V>>;

    #[inline(always)]
    fn handles_from_flat(
        _shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Handle<ViewPort<V>> {
        Handle::new(*indices.pop())
    }
}

// -- Leaf: RefViewPort<V> ---------------------------------------------------

impl<V: ValueView> InterfaceHandles for RefViewPort<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Handles<'a> = Handle<RefViewPort<V>>;
    type HandlesOwned = Handle<RefViewPort<V>>;

    #[inline(always)]
    fn handles_from_flat(
        _shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Handle<RefViewPort<V>> {
        Handle::new(*indices.pop())
    }
}

// -- Variadic leaf: RefViewPorts<V> -----------------------------------------

impl<V: ValueView> InterfaceHandles for RefViewPorts<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Handles<'a> = &'a [Handle<RefViewPort<V>>];
    type HandlesOwned = Box<[Handle<RefViewPort<V>>]>;

    #[inline]
    fn handles_from_flat(
        shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Self::HandlesOwned {
        let n = *shape.pop();
        (0..n).map(|_| Handle::new(*indices.pop())).collect()
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
/// goes the other way — a [`Handle`] is structural in its port type, so the
/// handle *argument* determines `Self` and `Self::Interface` normalizes forward.
/// Spelling [`Builder::push`](super::Builder::push)'s input as a free
/// `H: HandlesInterface` with `O: Segment<Inputs = H::Interface>` therefore lets
/// an operator's generics be **inferred from the wiring** — e.g.
/// `push(Op::new(..), handle)` with no turbofish.
pub trait HandlesInterface {
    /// The interface these handles wire; its inverse [`InterfaceHandles::Handles`]
    /// reconstitutes `Self`.
    type Interface: Interface;

    /// Write every leaf's node index into `indices` in tree-order, pushing each
    /// variadic ([`RefViewPorts`]) group's element count into `shape`. The walk-
    /// to-flat direction, owned wholly by this (handle-keyed) side.
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

// -- Leaf: Handle<ViewPort<V>> ----------------------------------------------

impl<V: ValueView> HandlesInterface for Handle<ViewPort<V>>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Interface = ViewPort<V>;

    #[inline(always)]
    fn to_vec(self, _shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        indices.push(self.index);
    }
}

// -- Leaf: Handle<RefViewPort<V>> -------------------------------------------

impl<V: ValueView> HandlesInterface for Handle<RefViewPort<V>>
where
    for<'a> V::View<'a>: Sync,
{
    type Interface = RefViewPort<V>;

    #[inline(always)]
    fn to_vec(self, _shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        indices.push(self.index);
    }
}

// -- Variadic leaf: &[Handle<RefViewPort<V>>] -------------------------------

impl<V: ValueView> HandlesInterface for &[Handle<RefViewPort<V>>]
where
    for<'a> V::View<'a>: Sync,
{
    type Interface = RefViewPorts<V>;

    #[inline]
    fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(self.len());
        indices.extend(self.iter().map(|h| h.index));
    }
}

// A reference to a fixed-size array of handles, e.g. `&[a, b, c]`. With a free
// handle type parameter there is no slice-typed parameter to unsize-coerce into,
// so the array case is supported explicitly (delegating to the slice logic).
impl<V: ValueView, const M: usize> HandlesInterface for &[Handle<RefViewPort<V>>; M]
where
    for<'a> V::View<'a>: Sync,
{
    type Interface = RefViewPorts<V>;

    #[inline]
    fn to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(M);
        indices.extend(self.iter().map(|h| h.index));
    }
}
