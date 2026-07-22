use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::{Interface, Operator};
use crate::ports::{ArrayPort, StripNotify};

/// In-place multi-input map: `Fn(views, &mut Array<SO, NO>) -> bool`.
pub struct ApplyInplace<I, SO: Scalar, const NO: usize, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    f: F,
    initial: Array<SO, NO>,
    _phantom: PhantomData<fn() -> I>,
}

impl<I, SO: Scalar, const NO: usize, F> ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    pub fn new(f: F, initial: Array<SO, NO>) -> Self {
        Self {
            f,
            initial,
            _phantom: PhantomData,
        }
    }
}

/// Runtime state for [`ApplyInplace`]: the function and the output buffer
/// (seeded from the initial value in `init`).
pub struct ApplyInplaceState<SO: Scalar, const NO: usize, F> {
    f: F,
    out: Array<SO, NO>,
}

impl<I, SO: Scalar, const NO: usize, F> Operator for ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    type Inputs = I;
    type Outputs = ArrayPort<SO, NO>;
    type Context = Instant;
    type State = ApplyInplaceState<SO, NO, F>;

    fn init(self, inputs: <I as Interface>::Values<'_>) -> Self::State {
        let mut out = self.initial;
        (self.f)(<I as StripNotify>::plain(inputs), &mut out);
        ApplyInplaceState { f: self.f, out }
    }

    fn compute<'a, 'b: 'a>(
        inputs: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        let notify = (state.f)(<I as StripNotify>::plain(inputs), &mut state.out);
        (notify, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: <I as Interface>::Values<'a>,
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, SO, NO>) {
        (false, state.out.view())
    }
}

/// [`apply`](fn@super::apply) writing into a reused output buffer; the closure
/// returns whether to notify.
pub fn apply_inplace<I, SO: Scalar, const NO: usize, F>(
    f: F,
    initial: Array<SO, NO>,
) -> ApplyInplace<I, SO, NO, F>
where
    I: StripNotify + 'static,
    F: for<'a> Fn(<I as StripNotify>::Plain<'a>, &mut Array<SO, NO>) -> bool
        + Send
        + Sync
        + 'static,
{
    ApplyInplace::new(f, initial)
}
