//! The new [`Operator`] trait, the [`Adapt`] bridge onto `flowgraph`, and the
//! shared [`Clock`].
//!
//! Operators are **pure** (no `timestamp` parameter), so `Adapt<O>` is a trivial
//! bool→notify wrapper with no per-operator state injection — operators are
//! directly `Adapt`-able. Event time is needed by exactly one operator
//! ([`Record`](super::Record)); it receives the [`Clock`] in its own state via
//! [`Scenario::add_record`](super::Scenario::add_record), so the clock is never
//! a universal dependency.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use flowgraph::typed::{Operator as FgOperator, Port, Ports};

use crate::Instant;

// ===========================================================================
// Clock — driver-advanced event time, held only by operators that need it.
// ===========================================================================

/// A clock the driver advances before each `stabilize`. The operators that stamp
/// event time ([`Record`](super::Record)) hold a clone in their own state and
/// read it via [`get`](Self::get). `Arc<AtomicI64>` is `Send + Sync`;
/// `Release`/`Acquire` make it self-synchronizing so a worker thread always
/// observes the latest `set`.
#[derive(Clone)]
pub struct Clock(Arc<AtomicI64>);

impl Clock {
    pub fn new() -> Self {
        Clock(Arc::new(AtomicI64::new(i64::MIN)))
    }

    /// Advance the clock. MUST be called only on the driver thread while no
    /// `stabilize` is in flight (i.e. between generations).
    pub fn set(&self, t: Instant) {
        self.0.store(t.as_nanos(), Ordering::Release);
    }

    pub fn get(&self) -> Instant {
        Instant::from_nanos(self.0.load(Ordering::Acquire))
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Operator — the (pure) TradingFlow operator contract, on flowgraph `Ports`.
// ===========================================================================

/// A synchronous computation node: reads typed inputs and writes a single typed
/// output, returning whether downstream should treat the output as newly
/// produced. Pure with respect to time — operators that need event time take it
/// as data (e.g. via the [`Clock`] in their state).
pub trait Operator: 'static {
    /// Input tree, e.g. `Port<Array<f64>>`, `(Port<A>, Port<B>)`, or `[Port<T>]`.
    type Inputs: Ports + ?Sized;
    /// Single output value type. Required `Send + Sync` (matches
    /// [`flowgraph::typed::Operator`]); every operator output/state is `Sync`.
    type Output: Send + Sync + 'static;
    /// Mutable per-run state.
    type State: Send + Sync + 'static;

    fn init(&self, inputs: <Self::Inputs as Ports>::Refs<'_>) -> (Self::State, Self::Output);

    /// Returns `true` iff this output should be treated as newly produced
    /// (downstream sees it as notified). `produced` is the per-input notify
    /// tree (the engine's name for the legacy `produced` bits).
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as Ports>::Refs<'_>,
        output: &mut Self::Output,
        produced: <Self::Inputs as Ports>::Notify<'_>,
    ) -> bool;
}

// ===========================================================================
// Adapt — the bridge: `O: Operator`  ==>  `Adapt<O>: flowgraph::typed::Operator`.
// ===========================================================================

/// Wraps an [`Operator`] so it can be pushed into a `flowgraph` graph: maps the
/// `bool` return onto the output notify flag. No clock, no extra state.
pub struct Adapt<O> {
    pub inner: O,
}

impl<O> Adapt<O> {
    pub fn new(inner: O) -> Self {
        Self { inner }
    }
}

impl<O: Operator> FgOperator for Adapt<O> {
    type Inputs = O::Inputs;
    type Outputs = Port<O::Output>;
    type State = O::State;

    fn init(&self, inputs: <O::Inputs as Ports>::Refs<'_>) -> (O::Output, O::State) {
        let (state, output) = self.inner.init(inputs);
        (output, state)
    }

    fn compute(
        inputs: <O::Inputs as Ports>::Refs<'_>,
        inputs_notify: <O::Inputs as Ports>::Notify<'_>,
        output: &mut O::Output,
        output_notify: &mut bool,
        state: &mut O::State,
    ) {
        // The TradingFlow propagation bool *is* the output notify flag.
        *output_notify = O::compute(state, inputs, output, inputs_notify);
    }
}
