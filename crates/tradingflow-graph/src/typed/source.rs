use std::marker::PhantomData;

use super::{Pass, Port, Segment};

/// A source node: no inputs, one output derived from node state.
pub struct Source<V: Pass, C>(V::Owned, PhantomData<C>);

impl<V: Pass, C> Source<V, C> {
    pub fn new(initial: V::Owned) -> Self {
        Self(initial, PhantomData)
    }
}

impl<V: Pass, C> Segment for Source<V, C>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
    C: Send + Sync + 'static,
{
    type Inputs = ();
    type Outputs = Port<V>;
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
        let owned: &'a V::Owned = state;
        (!is_first_run, V::borrow(owned))
    }
}
