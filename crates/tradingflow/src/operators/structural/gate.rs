//! `Gate` — view gate that honours the no-notify⟹unchanged contract.

use std::marker::PhantomData;

use crate::data::{Array, ArrayView, Instant, Scalar};
use crate::graph::typed::Operator;
use crate::ports::ArrayPort;

/// View gate: emits the input row as a `ViewPort`, notifying iff the input
/// notified AND the predicate holds — the row-cutoff that drops the all-NaN "no
/// data" cross-sections a dense panel emits for an idle stock.
///
/// The TradingFlow contract is that **an operator that does not notify must not
/// change its output value** (so any consumer may treat a non-notifying input
/// as its last notified value — the carry that
/// [`stack`](crate::operators::array::stack) relies on). A naive forwarder would break it: gating out a *notified* all-NaN row
/// while forwarding that row changes the value under `notify = false`. So
/// `Gate` retains the last passed row in owned state and re-presents a view of
/// it whenever it gates out or its input is silent. The retained buffer is
/// overwritten **in place** (no realloc) only on a pass — i.e. only when `Gate`
/// notifies — so a view stored by an out-of-cone consumer always reads the
/// frozen last-passed value. This makes `Gate`'s output a stable backing for
/// downstream zero-copy view chains.
pub struct Gate<T: Scalar, F, const N: usize>(pub F, pub PhantomData<T>);

/// Runtime state for [`Gate`]: the predicate plus the retained last-passed row,
/// which the `ViewPort` output borrows.
pub struct GateState<T: Scalar, F, const N: usize> {
    predicate: F,
    out: Array<T, N>,
}

impl<T: Scalar, F, const N: usize> Operator for Gate<T, F, N>
where
    F: for<'x> Fn(ArrayView<'x, T, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ArrayPort<T, N>;
    type Outputs = ArrayPort<T, N>;
    type Context = Instant;
    type State = GateState<T, F, N>;

    fn init(self, (_, view): (bool, ArrayView<'_, T, N>)) -> Self::State {
        GateState {
            predicate: self.0,
            // Seed the retained buffer with the faithful build-time row (so the
            // first view matches what `Split` lends).
            out: view.to_array(),
        }
    }

    fn compute<'a, 'b: 'a>(
        (notified, view): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        _: &Instant,
    ) -> (bool, ArrayView<'a, T, N>) {
        if notified && (state.predicate)(view) {
            // Pass: refresh the retained row in place (no realloc) and notify.
            state.out.assign(view);
            (true, state.out.view())
        } else {
            // Gate out (or upstream silent): re-present the unchanged retained
            // row under `notify = false` — the contract.
            (false, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

/// Like [`filter`](fn@super::filter), but re-presents the last passed row as a
/// stable [`ArrayPort`] view (the carry-safe view gate).
pub fn gate<T: Scalar, F, const N: usize>(predicate: F) -> Gate<T, F, N> {
    Gate(predicate, PhantomData)
}
