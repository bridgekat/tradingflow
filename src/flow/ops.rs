//! First batch of operators ported onto the new [`Operator`](super::Operator)
//! trait. Generic over [`Scalar`] where natural; `f64`-specialized for the
//! arithmetic / gating / slice / rolling ops to keep bounds tight (the generic
//! versions land with the bulk port).

use std::marker::PhantomData;

use flowgraph::typed::{Port, Ports};

use super::op::{Clock, Operator};
use crate::{Array, Scalar, Series};

// ---------------------------------------------------------------------------
// Const — externally-set constant / source cell.
// ---------------------------------------------------------------------------

/// 0-input source cell: never runs during `stabilize`; the driver mutates it
/// via `cell_mut` to inject events.
pub struct Const<T>(pub T);

impl<T: Clone + Send + Sync + 'static> Operator for Const<T> {
    type Inputs = ();
    type Output = T;
    type State = ();

    fn init(&self, _inputs: ()) -> ((), T) {
        ((), self.0.clone())
    }

    fn compute(_s: &mut (), _i: (), _o: &mut T, _p: ()) -> bool {
        true
    }
}

// (Elementwise arithmetic — Add/Subtract/Multiply/Divide/Min/Max/unary/Pow —
// lives in `super::arith`, ported from `operators::num::arithmetic`.)

// ---------------------------------------------------------------------------
// Filter — whole-array gate by predicate (the cutoff operator).
// ---------------------------------------------------------------------------

/// Passes the input through when the predicate holds, else drops it (returns
/// `false` → output un-notified → downstream gated off).
pub struct Filter<F>(pub F);

impl<F> Operator for Filter<F>
where
    F: Fn(&Array<f64>) -> bool + Clone + Send + Sync + 'static,
{
    type Inputs = Port<Array<f64>>;
    type Output = Array<f64>;
    type State = F;

    fn init(&self, inputs: &Array<f64>) -> (F, Array<f64>) {
        (self.0.clone(), inputs.clone())
    }

    fn compute(
        state: &mut F,
        inputs: &Array<f64>,
        output: &mut Array<f64>,
        _p: bool,
    ) -> bool {
        if state(inputs) {
            output.as_mut_slice().clone_from_slice(inputs.as_slice());
            true
        } else {
            false
        }
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

impl<T: Scalar> Operator for Record<T> {
    type Inputs = Port<Array<T>>;
    type Output = Series<T>;
    type State = Clock;

    fn init(&self, inputs: &Array<T>) -> (Clock, Series<T>) {
        (self.clock.clone(), Series::new(inputs.shape()))
    }

    fn compute(
        state: &mut Clock,
        inputs: &Array<T>,
        output: &mut Series<T>,
        _produced: bool,
    ) -> bool {
        output.push(state.get(), inputs.as_slice());
        true
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

impl<T: Scalar> Operator for Last<T> {
    type Inputs = Port<Series<T>>;
    type Output = Array<T>;
    type State = T;

    fn init(&self, inputs: &Series<T>) -> (T, Array<T>) {
        let shape = inputs.shape();
        let out = match inputs.last() {
            Some(last) => Array::from_vec(shape, last.to_vec()),
            None => Array::full(shape, self.fill.clone()),
        };
        (self.fill.clone(), out)
    }

    fn compute(
        state: &mut T,
        inputs: &Series<T>,
        output: &mut Array<T>,
        _p: bool,
    ) -> bool {
        match inputs.last() {
            Some(last) => output.as_mut_slice().clone_from_slice(last),
            None => {
                for v in output.as_mut_slice().iter_mut() {
                    *v = state.clone();
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Count — stateful per-tick counter (anti-corruption demonstrator).
// ---------------------------------------------------------------------------

/// Increments a counter every time it runs and emits the running count. Used to
/// prove gating advances state only when an input actually notifies.
pub struct Count;

impl Operator for Count {
    type Inputs = Port<Array<f64>>;
    type Output = Array<f64>;
    type State = i64;

    fn init(&self, _i: &Array<f64>) -> (i64, Array<f64>) {
        (0, Array::scalar(0.0))
    }

    fn compute(
        state: &mut i64,
        _i: &Array<f64>,
        out: &mut Array<f64>,
        _p: bool,
    ) -> bool {
        *state += 1;
        out.as_mut_slice()[0] = *state as f64;
        true
    }
}

// ---------------------------------------------------------------------------
// Clocked — clock-gated wrapper (exercises the ?Sized trailing-input path).
// ---------------------------------------------------------------------------

/// Prepends a leading `Port<C>` clock input; runs the inner operator only when
/// the clock notifies. `O::Inputs` may itself be `?Sized` (a slice).
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

impl<O: Operator, C: Send + Sync + 'static> Operator for Clocked<O, C> {
    type Inputs = (Port<C>, O::Inputs);
    type Output = O::Output;
    type State = O::State;

    fn init(
        &self,
        inputs: (&C, <O::Inputs as Ports>::Refs<'_>),
    ) -> (O::State, O::Output) {
        self.inner.init(inputs.1)
    }

    fn compute(
        state: &mut O::State,
        inputs: (&C, <O::Inputs as Ports>::Refs<'_>),
        output: &mut O::Output,
        produced: (bool, <O::Inputs as Ports>::Notify<'_>),
    ) -> bool {
        let (clock_fired, inner_produced) = produced;
        if !clock_fired {
            return false;
        }
        O::compute(state, inputs.1, output, inner_produced)
    }
}

// (Stack/StackSync/Concat/ConcatSync — axis-based slice-input reshape ops —
// live in `super::reshape`. Rolling operators live in `super::rolling`.)
