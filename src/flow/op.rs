//! Operator-contract helpers shared by the `flow` operators: the [`Clock`] and
//! the [`StripNotify`] refs helper.
//!
//! Since the `flowgraph` segment redesign, TradingFlow operators implement
//! [`flowgraph::typed::Operator`] (notify-gated; the common case) or
//! [`flowgraph::typed::Segment`] (custom gating, e.g. [`Clocked`](super::Clocked))
//! **directly** — there is no TradingFlow-side operator trait or `Adapt` bridge
//! any more, so operators compose with `flowgraph`'s combinators and the
//! `segment!` fusion macro as-is.
//!
//! Conventions shared by every operator in this module tree:
//!
//! * **Output storage lives in `State`.** `compute` writes the state-owned
//!   buffer and returns `(notify, &out)` references into it. State cells are
//!   boxed by the engine, so those references stay valid across generations
//!   (`passthrough` never reallocates).
//! * **The `init == true` call replicates the legacy `init(&self, inputs)`.**
//!   It only sizes/seeds the state and output from the build-time input values
//!   and returns `(false, &out)` — no per-tick side effect (no counter bump, no
//!   series append) may run on the build call.
//! * **`passthrough` returns `(false, &out)`** — the previous value,
//!   un-notified.
//!
//! Operators are **pure** with respect to time. Event time is needed by the few
//! operators that stamp it (e.g. [`Record`](super::Record)); they receive the
//! [`Clock`] in their own state, so the clock is never a universal dependency.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use flowgraph::typed::{Interface, Port, Ports};

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
// StripNotify — values-only view of an `Interface` refs tree.
// ===========================================================================

/// Maps an [`Interface`] refs tree (`(bool, &T)` leaves) onto its values-only
/// tree (`&T` leaves), dropping the notify flags. Lets closure operators
/// ([`Map`](super::Map) / [`Apply`](super::Apply)) keep their legacy closure
/// signatures, e.g. `Fn((&Array<f64>, &Array<f64>)) -> T` for a two-port input.
pub trait StripNotify: Interface {
    /// The values-only refs tree.
    type Values<'a>: Clone;

    /// Drop the notify flags from a refs tree.
    fn values<'a>(refs: Self::Refs<'a>) -> Self::Values<'a>;
}

impl<T: 'static> StripNotify for Port<T> {
    type Values<'a> = &'a T;

    #[inline(always)]
    fn values<'a>(refs: <Self as Interface>::Refs<'a>) -> Self::Values<'a> {
        refs.1
    }
}

impl StripNotify for () {
    type Values<'a> = ();

    #[inline(always)]
    fn values<'a>(_: <Self as Interface>::Refs<'a>) -> Self::Values<'a> {}
}

impl<T: 'static> StripNotify for Ports<T> {
    type Values<'a> = &'a [&'a T];

    #[inline(always)]
    fn values<'a>(refs: <Self as Interface>::Refs<'a>) -> Self::Values<'a> {
        refs.1
    }
}

macro_rules! impl_strip_notify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: StripNotify,)+> StripNotify for ($($T,)+) {
            type Values<'a> = ($($T::Values<'a>,)+);

            #[inline(always)]
            fn values<'a>(refs: Self::Refs<'a>) -> Self::Values<'a> {
                ( $( <$T as StripNotify>::values(refs.$idx), )+ )
            }
        }
    };
}

impl_strip_notify_for_tuple!(0: A);
impl_strip_notify_for_tuple!(0: A, 1: B);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_strip_notify_for_tuple!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
