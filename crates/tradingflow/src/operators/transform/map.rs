use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// Allocating map: applies `Fn(ArrayView<SI, NI>) -> Array<SO, NO>` each tick,
/// homing the result and lending a view of it.
pub struct Map<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> {
    f: F,
    _phantom: PhantomData<(SI, SO)>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Map`]: the function plus the output cell, seeded by
/// running the closure once on the build-time input in `init`.
pub struct MapState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F> Operator
    for Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = ArrayPort<SI, NI>;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = MapState<SO, NO, F>;

    fn init(self, (_, x): (bool, ArrayView<'_, SI, NI>)) -> Self::State {
        // Run the closure once on the build-time input to seed the output; the
        // initial render then goes through `passthrough` without notifying.
        let out = (self.f)(x);
        MapState { f: self.f, out }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        state.out = (state.f)(x);
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, SI, NI>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

/// Element-wise (well, whole-array) closure `ArrayView -> Array`.
pub fn map<SI: Scalar, const NI: usize, SO: Scalar, const NO: usize, F>(
    f: F,
) -> Map<SI, NI, SO, NO, F>
where
    F: for<'a> Fn(ArrayView<'a, SI, NI>) -> Array<SO, NO> + Send + Sync + 'static,
{
    Map::new(f)
}
