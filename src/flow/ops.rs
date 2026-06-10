//! Core gating / recording operators, implemented directly on
//! [`flowgraph::typed::Operator`] / [`Segment`]. (The legacy `Const` cell is
//! gone — source cells are `GraphBuilder::push_source` nodes now, registered
//! via [`Scenario::add_const`](super::Scenario::add_const).)

use std::marker::PhantomData;

use flowgraph::typed::{Interface, Operator, Port, Segment};

use super::op::Clock;
use crate::{Array, Scalar, Series};

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
    type Inputs = Port<Array<f64>>;
    type Outputs = Port<Array<f64>>;
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
// Record — append an Array stream into a Series, stamping with event time.
// ---------------------------------------------------------------------------

/// Records an `Array<T>` stream into a `Series<T>`, stamping each row with the
/// event time read from the [`Clock`] in its state. The one operator that needs
/// time — constructed via [`Scenario::add_record`](super::Scenario::add_record),
/// which supplies the driver's clock.
pub struct Record<T: Scalar> {
    clock: Clock,
    _p: PhantomData<T>,
}

impl<T: Scalar> Record<T> {
    pub fn new(clock: Clock) -> Self {
        Self {
            clock,
            _p: PhantomData,
        }
    }
}

/// Runtime state for [`Record`]: the clock plus the recorded series.
pub struct RecordState<T: Scalar> {
    clock: Clock,
    out: Series<T>,
}

impl<T: Scalar> Operator for Record<T> {
    type Inputs = Port<Array<T>>;
    type Outputs = Port<Series<T>>;
    type State = RecordState<T>;

    fn init(self) -> RecordState<T> {
        RecordState {
            clock: self.clock,
            out: Series::new(&[]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, x): (bool, &'a Array<T>),
        state: &'b mut RecordState<T>,
        init: bool,
    ) -> (bool, &'a Series<T>) {
        if init {
            state.out = Series::new(x.shape());
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
    type Inputs = Port<Series<T>>;
    type Outputs = Port<Array<T>>;
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
    type Inputs = Port<Array<f64>>;
    type Outputs = Port<Array<f64>>;
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

/// Prepends a leading `Port<C>` clock input; runs the inner operator's compute
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

impl<O: Operator, C: 'static> Segment for Clocked<O, C> {
    type Inputs = (Port<C>, O::Inputs);
    type Outputs = O::Outputs;
    type State = O::State;

    fn init(self) -> O::State {
        O::init(self.inner)
    }

    fn compute<'a, 'b: 'a>(
        ((clock_fired, _), rest): ((bool, &'a C), <O::Inputs as Interface>::Refs<'a>),
        state: &'b mut O::State,
        init: bool,
    ) -> <O::Outputs as Interface>::Refs<'a> {
        if init || clock_fired {
            O::compute(rest, state, init)
        } else {
            O::passthrough(rest, &*state)
        }
    }
}
