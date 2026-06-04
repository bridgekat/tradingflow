//! The new [`Operator`] trait, the [`Adapt`] bridge onto `flowgraph`, and the
//! shared [`Clock`] that threads event time.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use flowgraph::typed::{Operator as FgOperator, Port, Ports};

use crate::Instant;

// ===========================================================================
// Clock — time-as-data, shared between the driver and every operator's state.
// ===========================================================================

/// A driver-advanced clock cloned into every adapted operator's state.
///
/// The driver calls [`set`](Self::set) before each `stabilize`; operators that
/// stamp event time (e.g. [`Record`](super::Record)) read it via
/// [`get`](Self::get). `Arc<AtomicI64>` is `Send + Sync`, so it is a legal cell
/// payload. `Release`/`Acquire` make the clock self-synchronizing so a worker
/// thread always observes the latest `set`.
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

    /// Wrap an [`Operator`] into a `flowgraph`-registerable [`Adapt`] sharing
    /// this clock.
    pub fn op<O>(&self, inner: O) -> Adapt<O> {
        Adapt {
            inner,
            clock: self.clone(),
        }
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Operator — the TradingFlow operator contract, on flowgraph `Ports`.
// ===========================================================================

/// A synchronous computation node: reads typed inputs and writes a single typed
/// output, returning whether downstream should treat the output as newly
/// produced.
///
/// This mirrors [`crate::operator::Operator`] but over `flowgraph`'s
/// [`Ports`] (so `Inputs` and the `produced` tree are the engine's own types)
/// and with `Send + Sync` cell bounds.
pub trait Operator: 'static {
    /// Input tree, e.g. `Port<Array<f64>>`, `(Port<A>, Port<B>)`, or `[Port<T>]`.
    type Inputs: Ports + ?Sized;
    /// Single output value type. Required `Send + Sync` (matches
    /// [`flowgraph::typed::Operator`]); every operator output/state is `Sync`.
    type Output: Send + Sync + 'static;
    /// Mutable per-run state.
    type State: Send + Sync + 'static;

    fn init(
        &self,
        inputs: <Self::Inputs as Ports>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output);

    /// Returns `true` iff this output should be treated as newly produced
    /// (downstream sees it as notified). `produced` is the per-input notify
    /// tree (the engine's name for the legacy `produced` bits).
    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as Ports>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as Ports>::Notify<'_>,
    ) -> bool;
}

// ===========================================================================
// Adapt — the bridge: `O: Operator`  ==>  `Adapt<O>: flowgraph::typed::Operator`.
// ===========================================================================

/// Wraps an [`Operator`] so it can be pushed into a `flowgraph` graph. Maps the
/// `bool` return onto the output notify flag and threads time via the [`Clock`].
pub struct Adapt<O> {
    pub inner: O,
    pub clock: Clock,
}

impl<O: Operator> FgOperator for Adapt<O> {
    type Inputs = O::Inputs;
    type Outputs = Port<O::Output>;
    type State = (O::State, Clock);

    fn init(&self, inputs: <O::Inputs as Ports>::Refs<'_>) -> (O::Output, (O::State, Clock)) {
        let (state, output) = self.inner.init(inputs, self.clock.get());
        (output, (state, self.clock.clone()))
    }

    fn compute(
        inputs: <O::Inputs as Ports>::Refs<'_>,
        inputs_notify: <O::Inputs as Ports>::Notify<'_>,
        output: &mut O::Output,
        output_notify: &mut bool,
        state: &mut (O::State, Clock),
    ) {
        let ts = state.1.get();
        // The TradingFlow propagation bool *is* the output notify flag.
        *output_notify = O::compute(&mut state.0, inputs, output, ts, inputs_notify);
    }
}
