use std::marker::PhantomData;

use super::{Ref, Scalar, Segment, Value, ViewPort};

/// A by-value input source: no inputs, outputs [`ViewPort<V>`] borrowing a
/// view of the [`V::Owned`](Value::Owned) cell held in state. Poke the cell
/// via [`state_mut`](super::Graph::state_mut) — which dirties the node, so the
/// view is re-derived (and re-homed in the node's output scratch) before any
/// consumer in the cone reads it, even if the write moved the owned buffer.
/// A source runs only when poked, and then notifies — so *no-notify ⟹ output
/// unchanged* holds trivially. The alias [`Source<T>`] is the plain
/// `T`-by-value source (a `Scalar` view is its own owned form).
///
/// `C` is the graph-level context type — a source never reads it, so it stays
/// generic and is pinned by [`push_source`](super::Builder::push_source) to
/// the builder's (the `()` default only matters for spelled-out types).
pub struct ViewSource<V: Value, C>(V::Owned, PhantomData<C>);

impl<V: Value, C> ViewSource<V, C> {
    pub fn new(initial: V::Owned) -> Self {
        Self(initial, PhantomData)
    }
}

impl<V: Value, C> Segment for ViewSource<V, C>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
    C: Send + Sync + 'static,
{
    type Inputs = ();
    type Outputs = ViewPort<V>;
    type Context = C;
    type State = V::Owned;

    fn init(self) -> Self::State {
        self.0
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        _: &C,
        state: &'b mut V::Owned,
        is_first_run: bool,
    ) -> (bool, V::View<'a>) {
        // Reborrow the owned cell at the generation lifetime and borrow a
        // fresh view of it -- no lifetime erasure needed, unlike a stored view.
        let owned: &'a V::Owned = state;
        (!is_first_run, V::borrow(owned))
    }
}

/// Scalar by-value source: output [`Port<T>`](super::Port).
pub type Source<T, C> = ViewSource<Scalar<T>, C>;

/// Whole-value by-reference source: output [`RefPort<T>`](super::RefPort).
/// Owns the `T` in node state and lends `&T` per generation ([`Ref<T>`] over
/// [`ViewSource`]); poke via [`state_mut`](super::Graph::state_mut).
pub type RefSource<T, C> = ViewSource<Ref<T>, C>;
