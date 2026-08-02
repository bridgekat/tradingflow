use crate::data::{ArrayView, Instant};
use crate::graph::Segment;
use crate::ports::SignalPort;

/// Operator signature for [`quiet`] etc.
pub struct Quiet<const N: usize> {
    extents: [usize; N],
}

impl<const N: usize> Quiet<N> {
    pub fn new(extents: [usize; N]) -> Self {
        Self { extents }
    }
}

impl<const N: usize> Segment for Quiet<N> {
    type Inputs = ();
    type Outputs = SignalPort<N>;
    type Context = Instant;
    type State = [usize; N];

    fn init(self, _: ()) -> Self::State {
        self.extents
    }

    fn reset<'a, 'b: 'a>(_: (), extents: &'b mut [usize; N]) -> ArrayView<'a, bool, N> {
        ArrayView::full(*extents, &false)
    }

    fn compute<'a, 'b: 'a>(
        _: (),
        extents: &'b mut [usize; N],
        _: &Instant,
    ) -> ArrayView<'a, bool, N> {
        ArrayView::full(*extents, &false)
    }
}

/// A quiet signal array cell.
pub fn quiet<const N: usize>(
    extents: [usize; N],
) -> impl Segment<Inputs = (), Outputs = SignalPort<N>, Context = Instant, State = [usize; N]> {
    Quiet::new(extents)
}
