use std::marker::PhantomData;

use super::{FlatRead, Interface, Pass, Port, Ports};

/// Typed handle to a node in the graph.
pub struct NodeHandle<T> {
    node: usize,
    _phantom: PhantomData<fn() -> T>,
}

impl<S> NodeHandle<S> {
    pub(crate) fn new(node: usize) -> Self {
        Self {
            node,
            _phantom: PhantomData,
        }
    }

    /// The graph node index of the source.
    pub fn index(&self) -> usize {
        self.node
    }
}

impl<S> Clone for NodeHandle<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for NodeHandle<S> {}

/// Typed handle to a port in the graph.
pub struct PortHandle<V: Pass> {
    index: usize,
    _phantom: PhantomData<fn() -> V>,
}

impl<V: Pass> PortHandle<V> {
    pub(crate) fn new(index: usize) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// The graph port index of the source.
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<V: Pass> Clone for PortHandle<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Pass> Copy for PortHandle<V> {}

/// Type-level association from interfaces to handles.
pub trait InterfaceHandles: Interface {
    /// Maps interfaces to handles.
    /// The inverse map of [`HandlesInterface::Interface`].
    type Handles<'a>: Copy + HandlesInterface<Interface = Self>;

    /// The owned version of [`Handles`](Self::Handles).
    type HandlesOwned;

    /// Constructs handles from node indices laid out in tree order, with each
    /// variadic group's element count given in `shape`.
    fn handles_from_flat(
        shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Self::HandlesOwned;
}

impl InterfaceHandles for () {
    type Handles<'a> = ();
    type HandlesOwned = ();

    fn handles_from_flat(_: &mut FlatRead<usize>, _: &mut FlatRead<usize>) {}
}

macro_rules! impl_handles_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: InterfaceHandles,)+> InterfaceHandles for ($($T,)+) {
            type Handles<'a> = ($($T::Handles<'a>,)+);
            type HandlesOwned = ($($T::HandlesOwned,)+);

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

impl<V: Pass> InterfaceHandles for Port<V> {
    type Handles<'a> = PortHandle<V>;
    type HandlesOwned = PortHandle<V>;

    fn handles_from_flat(
        _shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> PortHandle<V> {
        PortHandle::new(*indices.pop())
    }
}

impl<V: Pass> InterfaceHandles for Ports<V> {
    type Handles<'a> = &'a [PortHandle<V>];
    type HandlesOwned = Box<[PortHandle<V>]>;

    fn handles_from_flat(
        shape: &mut FlatRead<usize>,
        indices: &mut FlatRead<usize>,
    ) -> Self::HandlesOwned {
        let n = *shape.pop();
        (0..n).map(|_| PortHandle::new(*indices.pop())).collect()
    }
}

/// Type-level association from handles to interfaces.
pub trait HandlesInterface {
    /// Maps handles to interfaces.
    /// The inverse map of [`InterfaceHandles::Handles`].
    type Interface: Interface;

    /// Writes every leaf's node index into `indices` in tree-order, pushing
    /// each variadic group's element count into `shape`.
    fn indices_to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>);
}

impl HandlesInterface for () {
    type Interface = ();

    fn indices_to_vec(self, _shape: &mut Vec<usize>, _indices: &mut Vec<usize>) {}
}

macro_rules! impl_handles_interface_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: HandlesInterface,)+> HandlesInterface for ($($T,)+) {
            type Interface = ($($T::Interface,)+);

            fn indices_to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
                $( self.$idx.indices_to_vec(shape, indices); )+
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

impl<V: Pass> HandlesInterface for PortHandle<V> {
    type Interface = Port<V>;

    fn indices_to_vec(self, _shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        indices.push(self.index);
    }
}

impl<V: Pass> HandlesInterface for &[PortHandle<V>] {
    type Interface = Ports<V>;

    fn indices_to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(self.len());
        indices.extend(self.iter().map(|h| h.index));
    }
}

/// Implementation needed for fixed-size arrays since `&[...]` literals are
/// treated as arrays by default.
impl<V: Pass, const M: usize> HandlesInterface for &[PortHandle<V>; M] {
    type Interface = Ports<V>;

    fn indices_to_vec(self, shape: &mut Vec<usize>, indices: &mut Vec<usize>) {
        shape.push(M);
        indices.extend(self.iter().map(|h| h.index));
    }
}
