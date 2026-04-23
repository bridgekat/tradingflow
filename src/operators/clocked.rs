//! Clock-gated operator transformer and simple resampler.
//!
//! [`Clocked<O, C>`] wraps any [`Operator`] `O` — including those with
//! `!Sized` `Inputs` such as slice-input operators — and prepends a single
//! clock input of node type `C`.  The inner operator runs only when the
//! clock produces; it reads the latest values from all its data inputs
//! regardless of whether they produced this cycle (time-series semantics).
//!
//! The clock's value is ignored — only its per-cycle `produced` bit is
//! consulted — so `C` can be any node type: `()` for a standard unit
//! clock, `Array<T>` to gate on another array node's ticks, `Series<T>`
//! for a series.  Type inference usually picks `C` up from the clock
//! handle's type at `add_operator` time, so call sites rarely need to
//! annotate it.
//!
//! # Input layout
//!
//! `Clocked<O, C>::Inputs = (Input<C>, O::Inputs)`.  The clock occupies
//! local position 0; O's inputs occupy local positions 1‥.  Because
//! `O::Inputs` is the trailing field of the tuple, it may be `!Sized`
//! (e.g. `[Input<Array<T>>]` for a [`Stack`](crate::operators::Stack)
//! operator).
//!
//! # Resample
//!
//! [`Resample<O, C>`] is the common special case `Clocked<Id<O>, C>` —
//! re-emit the data input's latest value on every clock tick.  The clock
//! (type `C`) and data (type `O`) can be any pair of node types: 9
//! combinations overall across `()` / `Array<T>` / `Series<T>`, with the
//! two dtypes independent of one another (the clock's value is never
//! read, so its element type never has to match the data's).  This is
//! the primitive to reach for when you need to align two records that
//! would otherwise tick at heterogeneous cadences — e.g. cross-sectional
//! features driven by irregular financial-report updates being recorded
//! alongside a trading-day-only returns target.

use std::marker::PhantomData;

use super::id::Id;
use crate::data::{Input, InputTypes, Instant};
use crate::operator::Operator;

/// Wraps an operator so it only fires when a leading clock input produces.
///
/// The clock's type parameter `C` is the *node type* at position 0 of
/// the composed input tree: `()` for a pure unit clock source, `Array<T>`
/// or `Series<T>` to gate on a data-producing node.  Either way, only
/// the clock's `produced` bit is consulted; its payload is ignored.
///
/// Putting the clock at the leading position means `O::Inputs` remains
/// in trailing position and may be `?Sized`.
pub struct Clocked<O, C = ()> {
    inner: O,
    _phantom: PhantomData<fn() -> C>,
}

impl<O, C> Clocked<O, C> {
    /// Create a clock-gated wrapper around `inner`.
    pub fn new(inner: O) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<O: Operator, C: Send + 'static> Operator for Clocked<O, C> {
    type State = O::State;
    /// Clock `Input<C>` at position 0 followed by all of O's inputs.
    /// `O::Inputs` may be `?Sized` because it is the trailing field.
    type Inputs = (Input<C>, O::Inputs);
    type Output = O::Output;

    fn init(
        self,
        inputs: (&C, <O::Inputs as InputTypes>::Refs<'_>),
        timestamp: Instant,
    ) -> (O::State, O::Output) {
        self.inner.init(inputs.1, timestamp)
    }

    fn compute(
        state: &mut O::State,
        inputs: (&C, <O::Inputs as InputTypes>::Refs<'_>),
        output: &mut O::Output,
        timestamp: Instant,
        produced: (bool, <O::Inputs as InputTypes>::Produced<'_>),
    ) -> bool {
        let (clock_fired, inner_produced) = produced;
        if !clock_fired {
            return false;
        }
        O::compute(state, inputs.1, output, timestamp, inner_produced)
    }
}

/// Emit `data`'s latest value on every tick of `clock`.
///
/// A thin newtype over `Clocked<Id<O>, C>`: the inner `Id<O>` forwards
/// the data input, so the output matches the data node at the moment
/// the clock fires.  Clock and data are independent node types — `O`
/// for the data/output side, `C` for the clock side — because the
/// clock's value is never read, only its produced bit is consulted.
///
/// # Input layout
///
/// `Resample<O, C>::Inputs = (Input<C>, Input<O>)` — clock at position 0,
/// data at position 1.  `Resample<O, C>::Output = O`.
pub struct Resample<O, C>(Clocked<Id<O>, C>)
where
    O: Clone + Send + 'static,
    C: Send + 'static;

impl<O, C> Resample<O, C>
where
    O: Clone + Send + 'static,
    C: Send + 'static,
{
    /// Create a resampler for data type `O` triggered by clock type `C`.
    pub fn new() -> Self {
        Self(Clocked::new(Id::new()))
    }
}

impl<O, C> Default for Resample<O, C>
where
    O: Clone + Send + 'static,
    C: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<O, C> Operator for Resample<O, C>
where
    O: Clone + Send + 'static,
    C: Send + 'static,
{
    type State = <Clocked<Id<O>, C> as Operator>::State;
    type Inputs = <Clocked<Id<O>, C> as Operator>::Inputs;
    type Output = <Clocked<Id<O>, C> as Operator>::Output;

    fn init(
        self,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        timestamp: Instant,
    ) -> (Self::State, Self::Output) {
        self.0.init(inputs, timestamp)
    }

    fn compute(
        state: &mut Self::State,
        inputs: <Self::Inputs as InputTypes>::Refs<'_>,
        output: &mut Self::Output,
        timestamp: Instant,
        produced: <Self::Inputs as InputTypes>::Produced<'_>,
    ) -> bool {
        <Clocked<Id<O>, C> as Operator>::compute(state, inputs, output, timestamp, produced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array;

    #[test]
    fn resample_array_array_same_dtype() {
        let clock = Array::scalar(0.0_f64);
        let data = Array::scalar(7.0_f64);
        type R = Resample<Array<f64>, Array<f64>>;
        let (mut s, mut o) = R::new().init((&clock, &data), Instant::MIN);
        // clock not produced: no emission.
        assert!(!R::compute(
            &mut s,
            (&clock, &data),
            &mut o,
            Instant::from_nanos(1),
            (false, true),
        ));
        // clock produced: emit data's latest value.
        let data2 = Array::scalar(42.0_f64);
        assert!(R::compute(
            &mut s,
            (&clock, &data2),
            &mut o,
            Instant::from_nanos(2),
            (true, false),
        ));
        assert_eq!(o.as_slice(), &[42.0]);
    }

    #[test]
    fn resample_array_array_cross_dtype() {
        // Clock dtype doesn't have to match data dtype — only the
        // clock's produced bit is consulted.
        let clock = Array::scalar(0_i32);
        let data = Array::scalar(3.15_f64);
        type R = Resample<Array<f64>, Array<i32>>;
        let (mut s, mut o) = R::new().init((&clock, &data), Instant::MIN);
        assert!(R::compute(
            &mut s,
            (&clock, &data),
            &mut o,
            Instant::from_nanos(1),
            (true, true),
        ));
        assert_eq!(o.as_slice(), &[3.15]);
    }

    #[test]
    fn resample_unit_clock_array_data() {
        let data = Array::scalar(9.0_f64);
        type R = Resample<Array<f64>, ()>;
        let (mut s, mut o) = R::new().init((&(), &data), Instant::MIN);
        assert!(!R::compute(
            &mut s,
            (&(), &data),
            &mut o,
            Instant::from_nanos(1),
            (false, true),
        ));
        assert!(R::compute(
            &mut s,
            (&(), &data),
            &mut o,
            Instant::from_nanos(2),
            (true, false),
        ));
        assert_eq!(o.as_slice(), &[9.0]);
    }

    #[test]
    fn resample_unit_unit_is_trigger_relay() {
        type R = Resample<(), ()>;
        let (mut s, mut o) = R::new().init((&(), &()), Instant::MIN);
        assert!(!R::compute(
            &mut s,
            (&(), &()),
            &mut o,
            Instant::from_nanos(1),
            (false, true),
        ));
        assert!(R::compute(
            &mut s,
            (&(), &()),
            &mut o,
            Instant::from_nanos(2),
            (true, true),
        ));
    }
}
