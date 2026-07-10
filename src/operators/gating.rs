//! Core gating / recording operators, implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`]. (The legacy `Const` cell is
//! gone — source cells are `Builder::push_source` nodes now, registered
//! via [`ScenarioExt::add_const`](crate::ScenarioExt::add_const).)

use std::marker::PhantomData;

use flowgraph::typed::{Interface, Operator, RefPort, Segment, ViewPort};

use super::op::{ArrayValue, Clock};
use crate::{Array, ArrayView, Retention, Scalar, Series};

// ---------------------------------------------------------------------------
// Filter — whole-array gate by predicate (the cutoff operator).
// ---------------------------------------------------------------------------

/// Passes the input through when the predicate holds, else drops it (emits
/// `notify = false` → downstream gated off, previous value retained).
pub struct Filter<F, const N: usize>(pub F);

/// Runtime state for [`Filter`]: the predicate plus the retained output.
pub struct FilterState<F, const N: usize> {
    predicate: F,
    out: Array<f64, N>,
}

impl<F, const N: usize> Operator for Filter<F, N>
where
    F: for<'x> Fn(ArrayView<'x, f64, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ViewPort<ArrayValue<f64, N>>;
    type Outputs = ViewPort<ArrayValue<f64, N>>;
    type State = FilterState<F, N>;

    fn init(self) -> Self::State {
        FilterState {
            predicate: self.0,
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, f64, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, N>) {
        if init {
            state.out = x.to_array();
            return (false, state.out.view());
        }
        if (state.predicate)(x) {
            let xs = x.to_contiguous();
            state.out.assign(&xs);
            (true, state.out.view())
        } else {
            (false, state.out.view())
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, f64, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Gate — view gate that honours the no-notify⟹unchanged contract.
// ---------------------------------------------------------------------------

/// View gate: emits the input row as a `ViewPort`, notifying iff the input
/// notified AND the predicate holds — the row-cutoff that drops the all-NaN "no
/// data" cross-sections a dense panel emits for an idle stock.
///
/// The TradingFlow contract is that **an operator that does not notify must not
/// change its output value** (so any consumer may treat a non-notifying input
/// as its last notified value — the carry that [`Stack`](super::Stack) relies
/// on). A naive forwarder would break it: gating out a *notified* all-NaN row
/// while forwarding that row changes the value under `notify = false`. So
/// `Gate` retains the last passed row in owned state and re-presents a view of
/// it whenever it gates out or its input is silent. The retained buffer is
/// overwritten **in place** (no realloc) only on a pass — i.e. only when `Gate`
/// notifies — so a view stored by an out-of-cone consumer always reads the
/// frozen last-passed value. This makes `Gate`'s output a stable backing for
/// downstream zero-copy view chains.
pub struct Gate<F, const N: usize>(pub F);

/// Runtime state for [`Gate`]: the predicate plus the retained last-passed row,
/// which the `ViewPort` output borrows.
pub struct GateState<F, const N: usize> {
    predicate: F,
    out: Array<f64, N>,
}

impl<F, const N: usize> Operator for Gate<F, N>
where
    F: for<'x> Fn(ArrayView<'x, f64, N>) -> bool + Send + Sync + 'static,
{
    type Inputs = ViewPort<ArrayValue<f64, N>>;
    type Outputs = ViewPort<ArrayValue<f64, N>>;
    type State = GateState<F, N>;

    fn init(self) -> Self::State {
        GateState {
            predicate: self.0,
            out: Array::zeros([0; N]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, view): (bool, ArrayView<'a, f64, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, N>) {
        if init {
            // Seed the retained buffer with the faithful build-time row (so the
            // first view matches what `Split` lends), but do not notify.
            state.out = view.to_array();
            return (false, state.out.view());
        }
        if notified && (state.predicate)(view) {
            // Pass: refresh the retained row in place (no realloc) and notify.
            let xs = view.to_contiguous();
            state.out.assign(&xs);
            (true, state.out.view())
        } else {
            // Gate out (or upstream silent): re-present the unchanged retained
            // row under `notify = false` — the contract.
            (false, state.out.view())
        }
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, f64, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Record — append an Array stream into a Series, stamping with event time.
// ---------------------------------------------------------------------------

/// Records an `Array<T, N>` stream into a `Series<T, N>`, stamping each row with
/// the event time read from the [`Clock`] in its state. The one operator that
/// needs time — constructed via the [`record`](super::record) /
/// [`record_bounded`](super::record_bounded) formula constructors, which take
/// the driver's clock as a parameter.
///
/// An optional [`Retention`] bound (via [`with_retention`](Self::with_retention)
/// / [`record_bounded`](super::record_bounded)) caps the recorded history.
pub struct Record<T: Scalar, const N: usize> {
    clock: Clock,
    retention: Retention,
    _p: PhantomData<T>,
}

impl<T: Scalar, const N: usize> Record<T, N> {
    /// An unbounded record (retains full history).
    pub fn new(clock: Clock) -> Self {
        Self::with_retention(clock, Retention::UNBOUNDED)
    }

    /// A record whose `Series` keeps only the history within `retention`.
    pub fn with_retention(clock: Clock, retention: Retention) -> Self {
        Self {
            clock,
            retention,
            _p: PhantomData,
        }
    }
}

/// Runtime state for [`Record`]: the clock, the retention bound, plus the
/// recorded series.
pub struct RecordState<T: Scalar, const N: usize> {
    clock: Clock,
    retention: Retention,
    out: Series<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Record<T, N> {
    type Inputs = ViewPort<ArrayValue<T, N>>;
    type Outputs = RefPort<Series<T, N>>;
    type State = RecordState<T, N>;

    fn init(self) -> Self::State {
        RecordState {
            clock: self.clock,
            retention: self.retention,
            out: Series::new([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, ArrayView<'a, T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, &'a Series<T, N>) {
        if init {
            state.out = Series::with_retention(x.extents(), state.retention);
            return (false, &state.out);
        }
        let t = state
            .clock
            .get()
            .expect("Record: event time unset (the driver sets it before each stabilize)");
        state.out.push_view(t, &x);
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, T, N>),
        state: &'b Self::State,
    ) -> (bool, &'a Series<T, N>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Last — most recent element of a Series as an Array.
// ---------------------------------------------------------------------------

/// Extracts the most recent element of a `Series<T, N>` as a rank-`N`
/// [`ArrayView`], substituting `fill` when the series is empty.
pub struct Last<T: Scalar, const N: usize> {
    fill: T,
}

impl<T: Scalar, const N: usize> Last<T, N> {
    pub fn new(fill: T) -> Self {
        Self { fill }
    }
}

/// Runtime state for [`Last`]: the fill value plus the output buffer.
pub struct LastState<T: Scalar, const N: usize> {
    fill: T,
    out: Array<T, N>,
}

impl<T: Scalar, const N: usize> Operator for Last<T, N> {
    type Inputs = RefPort<Series<T, N>>;
    type Outputs = ViewPort<ArrayValue<T, N>>;
    type State = LastState<T, N>;

    fn init(self) -> Self::State {
        LastState {
            fill: self.fill.clone(),
            out: Array::zeros([0; N]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, &'a Series<T, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, T, N>) {
        if init {
            let ext = series.extents();
            state.out = match series.last() {
                Some(last) => Array::from_vec(ext, last.to_vec()),
                None => Array::full(ext, state.fill.clone()),
            };
            return (false, state.out.view());
        }
        match series.last() {
            Some(last) => state.out.as_mut_slice().clone_from_slice(last),
            None => {
                for v in state.out.as_mut_slice().iter_mut() {
                    *v = state.fill.clone();
                }
            }
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Series<T, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, T, N>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Count — stateful per-tick counter (anti-corruption demonstrator).
// ---------------------------------------------------------------------------

/// Increments a counter every time it runs and emits the running count (a
/// rank-0 scalar). Used to prove gating advances state only when an input
/// actually notifies.
pub struct Count<const N: usize>;

/// Runtime state for [`Count`]: the counter plus the scalar output buffer.
pub struct CountState {
    count: i64,
    out: Array<f64, 0>,
}

impl<const N: usize> Operator for Count<N> {
    type Inputs = ViewPort<ArrayValue<f64, N>>;
    type Outputs = ViewPort<ArrayValue<f64, 0>>;
    type State = CountState;

    fn init(self) -> Self::State {
        CountState {
            count: 0,
            out: Array::scalar(0.0),
        }
    }

    fn compute<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b mut Self::State,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        if init {
            return (false, state.out.view());
        }
        state.count += 1;
        state.out.as_mut_slice()[0] = state.count as f64;
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, N>),
        state: &'b Self::State,
    ) -> (bool, ArrayView<'a, f64, 0>) {
        (false, state.out.view())
    }
}

// ---------------------------------------------------------------------------
// Clocked — clock-gated wrapper.
// ---------------------------------------------------------------------------

/// Prepends a leading `RefPort<C>` clock input; runs the inner operator's compute
/// path only when the clock notifies, else the inner passthrough. Implements
/// [`Segment`] directly because its gate ignores the data inputs' notify bits.
pub struct Clocked<O, C = ()> {
    inner: O,
    _c: PhantomData<fn() -> C>,
}

impl<O, C> Clocked<O, C> {
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            _c: PhantomData,
        }
    }
}

// Manual `Clone` (avoids an unnecessary `C: Clone` bound — `C` is phantom).
impl<O: Clone, C> Clone for Clocked<O, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _c: PhantomData,
        }
    }
}

impl<O: Operator, C: Sync + 'static> Segment for Clocked<O, C> {
    type Inputs = (RefPort<C>, O::Inputs);
    type Outputs = O::Outputs;
    type State = O::State;

    fn init(self) -> O::State {
        O::init(self.inner)
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), rest): ((bool, &'a C), <O::Inputs as Interface>::Values<'a>),
        state: &'b mut O::State,
        init: bool,
    ) -> <O::Outputs as Interface>::Values<'a> {
        if init || clock_fired {
            O::compute(rest, state, init)
        } else {
            O::passthrough(rest, &*state)
        }
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Pass the input through iff `predicate` holds, else drop the tick (emitting
/// `notify = false`). The cutoff operator: a dropped tick suppresses every
/// downstream side effect, including a [`Record`] append.
pub fn filter<F, const N: usize>(predicate: F) -> Filter<F, N> {
    Filter(predicate)
}

/// Like [`filter`], but re-presents the last passed row as a stable
/// [`ViewPort`] view (the carry-safe view gate).
pub fn gate<F, const N: usize>(predicate: F) -> Gate<F, N> {
    Gate(predicate)
}

/// The most recent element of a [`Series`] as an array view, `fill` when empty.
pub fn last<T: Scalar, const N: usize>(fill: T) -> Last<T, N> {
    Last::new(fill)
}

/// Count the notified ticks seen so far.
pub fn count<const N: usize>() -> Count<N> {
    Count
}

/// Prepend a leading clock port to `inner`, running its compute path only when
/// the clock notifies.
pub fn clocked<O, C>(inner: O) -> Clocked<O, C> {
    Clocked::new(inner)
}
