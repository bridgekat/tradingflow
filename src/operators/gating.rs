//! Core gating / recording operators, implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`]. (The legacy `Const` cell is
//! gone — source cells are `GraphBuilder::push_source` nodes now, registered
//! via [`Scenario::add_const`](super::Scenario::add_const).)

use std::marker::PhantomData;

use flowgraph::typed::{Interface, Operator, RefPort, RefViewPort, Segment, ViewPort};

use super::op::{ArrayValue, Clock};
use crate::{Array, ArraySlice, Retention, Scalar, Series};

// ---------------------------------------------------------------------------
// Filter — whole-array gate by predicate (the cutoff operator).
// ---------------------------------------------------------------------------

/// Passes the input through when the predicate holds, else drops it (emits
/// `notify = false` → downstream gated off, previous value retained).
pub struct Filter<F>(pub F);

/// Runtime state for [`Filter`]: the predicate plus the retained output.
pub struct FilterState<F> {
    predicate: F,
    out: Array<f64>,
}

impl<F> Operator for Filter<F>
where
    F: Fn(&Array<f64>) -> bool + Send + Sync + 'static,
{
    type Inputs = RefPort<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
    type State = FilterState<F>;

    fn init(self) -> FilterState<F> {
        FilterState {
            predicate: self.0,
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a Array<f64>),
        state: &'b mut FilterState<F>,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            state.out = x.clone();
            return (false, &state.out);
        }
        if (state.predicate)(x) {
            state.out.as_mut_slice().clone_from_slice(x.as_slice());
            (true, &state.out)
        } else {
            (false, &state.out)
        }
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<f64>),
        state: &'b FilterState<F>,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Gate — view gate that honours the no-notify⟹unchanged contract.
// ---------------------------------------------------------------------------

/// View gate: emits the input [`ArraySlice`] as a `ViewPort`, notifying iff the
/// input notified AND the predicate holds — the row-cutoff that drops the
/// all-NaN "no data" cross-sections a dense panel emits for an idle stock.
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
pub struct Gate<F>(pub F);

/// Runtime state for [`Gate`]: the predicate plus the retained last-passed row,
/// which the `ViewPort` output borrows.
pub struct GateState<F> {
    predicate: F,
    out: Array<f64>,
}

impl<F> Operator for Gate<F>
where
    F: for<'x> Fn(ArraySlice<'x, f64>) -> bool + Send + 'static,
{
    type Inputs = RefViewPort<ArrayValue<f64>>;
    type Outputs = ViewPort<ArrayValue<f64>>;
    type State = GateState<F>;

    fn init(self) -> GateState<F> {
        GateState {
            predicate: self.0,
            out: Array::zeros(&[0]),
        }
    }

    #[inline(always)]
    fn compute<'a, 'b: 'a>(
        (notified, view): (bool, &'a ArraySlice<'a, f64>),
        state: &'b mut GateState<F>,
        init: bool,
    ) -> (bool, ArraySlice<'a, f64>) {
        if init {
            // Seed the retained buffer with the faithful build-time row (so the
            // first view matches what `Split` lends), but do not notify.
            state.out = view.to_array();
            return (false, ArraySlice::from(&state.out));
        }
        if notified && (state.predicate)(*view) {
            // Pass: refresh the retained row in place (no realloc) and notify.
            state.out.assign(view.as_slice());
            (true, ArraySlice::from(&state.out))
        } else {
            // Gate out (or upstream silent): re-present the unchanged retained
            // row under `notify = false` — the contract.
            (false, ArraySlice::from(&state.out))
        }
    }

    #[inline(always)]
    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a ArraySlice<'a, f64>),
        state: &'b GateState<F>,
    ) -> (bool, ArraySlice<'a, f64>) {
        (false, ArraySlice::from(&state.out))
    }
}

// ---------------------------------------------------------------------------
// Record — append an Array stream into a Series, stamping with event time.
// ---------------------------------------------------------------------------

/// Records an `Array<T>` stream into a `Series<T>`, stamping each row with the
/// event time read from the [`Clock`] in its state. The one operator that needs
/// time — constructed via [`Scenario::add_record`](super::Scenario::add_record),
/// which supplies the driver's clock.
///
/// An optional [`Retention`] bound (via [`with_retention`](Self::with_retention)
/// / [`Scenario::add_record_retained`](super::Scenario::add_record_retained))
/// caps the recorded history: the recorded `Series` drops its oldest rows once
/// they fall outside the bound, keeping memory bounded for records whose
/// consumers only look back a fixed window. The default is unbounded (full
/// history), matching the original behaviour.
pub struct Record<T: Scalar> {
    clock: Clock,
    retention: Retention,
    _p: PhantomData<T>,
}

impl<T: Scalar> Record<T> {
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
pub struct RecordState<T: Scalar> {
    clock: Clock,
    retention: Retention,
    out: Series<T>,
}

impl<T: Scalar> Operator for Record<T> {
    type Inputs = RefPort<Array<T>>;
    type Outputs = RefPort<Series<T>>;
    type State = RecordState<T>;

    fn init(self) -> RecordState<T> {
        RecordState {
            clock: self.clock,
            retention: self.retention,
            out: Series::new(&[]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a Array<T>),
        state: &'b mut RecordState<T>,
        init: bool,
    ) -> (bool, &'a Series<T>) {
        if init {
            state.out = Series::with_retention(x.shape(), state.retention);
            return (false, &state.out);
        }
        state.out.push(state.clock.get(), x.as_slice());
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<T>),
        state: &'b RecordState<T>,
    ) -> (bool, &'a Series<T>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Last — most recent element of a Series as an Array.
// ---------------------------------------------------------------------------

/// Extracts the most recent element of a `Series<T>` as an `Array<T>`,
/// substituting `fill` when the series is empty.
pub struct Last<T: Scalar> {
    fill: T,
}

impl<T: Scalar> Last<T> {
    pub fn new(fill: T) -> Self {
        Self { fill }
    }
}

/// Runtime state for [`Last`]: the fill value plus the output buffer.
pub struct LastState<T: Scalar> {
    fill: T,
    out: Array<T>,
}

impl<T: Scalar> Operator for Last<T> {
    type Inputs = RefPort<Series<T>>;
    type Outputs = RefPort<Array<T>>;
    type State = LastState<T>;

    fn init(self) -> LastState<T> {
        LastState {
            fill: self.fill.clone(),
            out: Array::zeros(&[0]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, series): (bool, &'a Series<T>),
        state: &'b mut LastState<T>,
        init: bool,
    ) -> (bool, &'a Array<T>) {
        if init {
            let shape = series.shape();
            state.out = match series.last() {
                Some(last) => Array::from_vec(shape, last.to_vec()),
                None => Array::full(shape, state.fill.clone()),
            };
            return (false, &state.out);
        }
        match series.last() {
            Some(last) => state.out.as_mut_slice().clone_from_slice(last),
            None => {
                for v in state.out.as_mut_slice().iter_mut() {
                    *v = state.fill.clone();
                }
            }
        }
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Series<T>),
        state: &'b LastState<T>,
    ) -> (bool, &'a Array<T>) {
        (false, &state.out)
    }
}

// ---------------------------------------------------------------------------
// Count — stateful per-tick counter (anti-corruption demonstrator).
// ---------------------------------------------------------------------------

/// Increments a counter every time it runs and emits the running count. Used to
/// prove gating advances state only when an input actually notifies.
pub struct Count;

/// Runtime state for [`Count`]: the counter plus the output buffer.
pub struct CountState {
    count: i64,
    out: Array<f64>,
}

impl Operator for Count {
    type Inputs = RefPort<Array<f64>>;
    type Outputs = RefPort<Array<f64>>;
    type State = CountState;

    fn init(self) -> CountState {
        CountState {
            count: 0,
            out: Array::scalar(0.0),
        }
    }

    fn compute<'a, 'b: 'a>(
        _: (bool, &'a Array<f64>),
        state: &'b mut CountState,
        init: bool,
    ) -> (bool, &'a Array<f64>) {
        if init {
            return (false, &state.out);
        }
        state.count += 1;
        state.out.as_mut_slice()[0] = state.count as f64;
        (true, &state.out)
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, &'a Array<f64>),
        state: &'b CountState,
    ) -> (bool, &'a Array<f64>) {
        (false, &state.out)
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
