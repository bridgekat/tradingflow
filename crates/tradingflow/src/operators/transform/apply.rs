use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::{Interface, Operator};
use crate::ports::{ArrayPort, StripNotify};

/// Allocating multi-input map: `Fn(views) -> Array<SO, NO>`, where `views` is
/// the values-only views tree of the input interface `I` (e.g.
/// `(ArrayView<f64, 2>, ArrayView<f64, 2>)` for two `ViewPort` ports — see
/// [`StripNotify`]).
pub struct Apply<I, SO: Scalar, const NO: usize, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    f: F,
    _phantom: PhantomData<fn() -> (I, SO)>,
}

impl<I, SO: Scalar, const NO: usize, F> Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`Apply`]: the function plus the output cell, seeded by
/// running the closure once on the build-time inputs in `init`.
pub struct ApplyState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<I, SO: Scalar, const NO: usize, F> Operator for Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    type Inputs = I;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = ApplyState<SO, NO, F>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> Self::State {
        // Run the closure once on the build-time inputs to seed the output; the
        // initial render then goes through `passthrough` without notifying.
        let out = (self.f)(<I as StripNotify>::plain(inputs));
        ApplyState { f: self.f, out }
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        state.out = (state.f)(<I as StripNotify>::plain(inputs));
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

/// A closure over a whole input *tree* (the multi-port [`map`](fn@super::map));
/// the input interface `I` is inferred from the wiring.
pub fn apply<I, SO: Scalar, const NO: usize, F>(f: F) -> Apply<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>) -> Array<SO, NO> + Send + Sync + 'static,
{
    Apply::new(f)
}
