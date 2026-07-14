use std::marker::PhantomData;

use super::{RefViewPort, Scalar, Segment, ValueView, ViewPort};

/// A by-value input source: no inputs, outputs [`ViewPort<V>`] carrying the
/// poked value itself. The alias [`Source<T>`] is the plain `T`-by-value source.
///
/// `C` is the graph-level context type — a source never reads it, so it stays
/// generic and is pinned by [`push_source`](super::Builder::push_source) to
/// the builder's (the `()` default only matters for spelled-out types).
pub struct ViewSource<V: ValueView, C = ()>(V::View<'static>, PhantomData<C>);

impl<V: ValueView, C> ViewSource<V, C> {
    pub fn new(initial: V::View<'static>) -> Self {
        Self(initial, PhantomData)
    }
}

impl<V: ValueView, C> Segment for ViewSource<V, C>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
    C: Send + Sync + 'static,
{
    type Inputs = ();
    type Outputs = ViewPort<V>;
    type Context = C;
    type State = V::View<'static>;

    fn init(self) -> Self::State {
        self.0
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        _: &C,
        state: &'b mut V::View<'static>,
        is_first_run: bool,
    ) -> (bool, V::View<'a>) {
        // SAFETY: covariance + `Copy`: read the `'static` view back at the
        // shorter `'a` (same layout); `Copy` means no double-ownership.
        let view = unsafe { *std::ptr::from_ref::<V::View<'static>>(state).cast::<V::View<'a>>() };
        (!is_first_run, view)
    }
}

/// Scalar by-value source: output [`Port<T>`](super::Port).
pub type Source<T, C = ()> = ViewSource<Scalar<T>, C>;

/// A by-reference input source: no inputs, outputs `RefViewPort<V>` whose `&V::View`
/// points at the poked value held in state. Poke via `state_mut`.
///
/// Holds the `'static` instantiation of the view in state (for [`Scalar<T>`]
/// that is just `T`); the alias [`RefSource<T>`] is the plain `&T` source.
/// `C` is the graph-level context type, generic as for [`ViewSource`].
pub struct RefViewSource<V: ValueView, C = ()>(V::View<'static>, PhantomData<C>);

impl<V: ValueView, C> RefViewSource<V, C> {
    pub fn new(initial: V::View<'static>) -> Self {
        Self(initial, PhantomData)
    }
}

impl<V: ValueView, C> Segment for RefViewSource<V, C>
where
    for<'a> V::View<'a>: Send + Sync,
    C: Send + Sync + 'static,
{
    type Inputs = ();
    type Outputs = RefViewPort<V>;
    type Context = C;
    type State = V::View<'static>;

    fn init(self) -> Self::State {
        self.0
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        _: &C,
        state: &'b mut V::View<'static>,
        is_first_run: bool,
    ) -> (bool, &'a V::View<'a>) {
        // SAFETY: covariance (the `ValueView` contract): `&V::View<'static>` is
        // usable as `&V::View<'a>` -- same layout, the lifetime only shortens.
        let view = unsafe { &*std::ptr::from_ref::<V::View<'static>>(state).cast::<V::View<'a>>() };
        (!is_first_run, view)
    }
}

/// Scalar by-reference source: output [`RefPort<T>`](super::RefPort).
pub type RefSource<T, C = ()> = RefViewSource<Scalar<T>, C>;
