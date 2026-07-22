use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// In-place map: `Fn(ArrayView<SI, NI>, &mut Array<SO, NO>) -> bool`; the bool
/// controls propagation.
pub struct MapInplace<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> {
    f: F,
    initial: Array<SO, NO>,
    _phantom: PhantomData<SI>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    pub fn new(f: F, initial: Array<SO, NO>) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`MapInplace`]: the function and the output buffer
/// (seeded from the initial value in `init`).
pub struct MapInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<SI, NI>;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = MapInplaceState<SO, NO, F>;

    fn init(self, (_, x): (bool, ArrayView<'_, SI, NI>)) -> Self::State {
        let mut out = self.initial;
        (self.f)(x, &mut out);
        MapInplaceState { f: self.f, out }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        let notify = (state.f)(x, &mut state.out);
        (notify, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

/// [`map`](fn@super::map) writing into a reused output buffer; the closure
/// returns whether to notify.
pub fn map_inplace<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F>(
    f: F,
    initial: Array<SO, NO>,
) -> MapInplace<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>, &mut Array<SO, NO>) -> bool + Send + Sync + 'static,
{
    MapInplace::new(f, initial)
}
