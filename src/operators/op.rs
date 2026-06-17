//! Operator-contract helpers shared by TradingFlow operators: the [`Clock`],
//! the [`ArrayValue`] view kind, and the [`StripNotify`] payload helper.
//!
//! TradingFlow operators implement [`flowgraph::typed::Operator`]
//! (notify-gated; the common case) or [`flowgraph::typed::Segment`] (custom
//! gating, e.g. [`Clocked`](super::Clocked)) **directly** — there is no
//! TradingFlow-side operator trait or bridge, so operators compose with
//! `flowgraph`'s combinators and the `segment!` fusion macro as-is.
//!
//! # The array currency
//!
//! Array-shaped edges all carry a borrowed, strided
//! [`ArrayView<'a, T, N>`](crate::ArrayView) by value through a
//! `ViewPort<ArrayValue<T, N>>` leaf — one currency for owned-buffer outputs and
//! zero-copy slices alike. A compute operator homes its output buffer in `State`
//! (an owned [`Array<T, N>`](crate::Array)) and returns a `ViewPort` view of it;
//! a slicing operator ([`SliceView`](super::SliceView)) re-derives a strided view
//! of its input by value, needing neither in-state shape nor an arena. Edges that
//! fan a single buffer into / out of `N` views ([`Split`](super::Split) outputs,
//! [`StackView`](super::StackView) inputs) use the by-reference
//! `RefViewPorts<ArrayValue<T, N>>` kind, whose `N` views are homed in a
//! per-generation arena.
//!
//! Conventions shared by every operator in this module tree:
//!
//! * **THE CONTRACT — *no-notify ⟹ output unchanged* (the load-bearing
//!   invariant of the whole engine).** Whenever an operator chooses **not** to
//!   notify, its output value MUST NOT change. Concretely: a `compute` that
//!   returns `(false, …)` must leave its output byte-identical to the value it
//!   last emitted under `(true, …)`, and `passthrough` re-emits that value
//!   verbatim. This is precisely what lets **any** consumer treat a
//!   non-notifying input as its last notified value — the carry that
//!   [`Stack`](super::Stack) / [`StackView`](super::StackView) depend on, and
//!   what makes a pure forwarding operator automatically correct. The duty
//!   falls on the **producer**, never the consumer: an operator that drops
//!   notifications while its input value changes (e.g. [`Gate`](super::Gate)'s
//!   row cutoff) MUST retain the last passed value in owned state and
//!   re-present it — it may never forward the dropped value under
//!   `notify = false`. A carry reader trusts this *universally*, so a single
//!   violating producer silently corrupts it; honour it in every operator.
//! * **Output storage lives in `State`** (owned buffers). Because array buffers
//!   are overwritten in place and never reallocated after the build call, a
//!   `ViewPort` view returned into `&'b mut State` stays valid across
//!   generations (`passthrough` re-lends the same buffer).
//! * **The `init == true` call replicates the legacy `init(&self, inputs)`.**
//!   It only sizes/seeds the state and output from the build-time input values
//!   and returns `(false, …)` — no per-tick side effect (no counter bump, no
//!   series append) may run on the build call.
//! * **`passthrough` returns `(false, …)`** — the previous value, un-notified.
//!
//! Operators are **pure** with respect to time. Event time is needed by the few
//! operators that stamp it (e.g. [`Record`](super::Record)); they receive the
//! [`Clock`] in their own state, so the clock is never a universal dependency.

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use flowgraph::typed::{Interface, RefViewPort, RefViewPorts, ValueView, ViewPort};

use crate::{ArrayView, Instant, Scalar};

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
// ArrayValue — the `ArrayView` view kind for flowgraph ports.
// ===========================================================================

/// [`ValueView`] kind passing a borrowed [`ArrayView<'a, T, N>`](crate::ArrayView)
/// across interfaces: `ViewPort<ArrayValue<T, N>>` /
/// `RefViewPort<ArrayValue<T, N>>` / `RefViewPorts<ArrayValue<T, N>>` leaves
/// carry the strided view with the engine's per-generation lifetime — fully
/// borrow-checked zero-copy edges. The rank `N` is compile-time; the operator's
/// generic rank is inferred from the input handle types at `add_operator`.
pub struct ArrayValue<T, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]` plus plain `usize`s, so it
// is covariant in `'a` — the only `ValueView` obligation.
unsafe impl<T: Scalar, const N: usize> ValueView for ArrayValue<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
}

// ===========================================================================
// StripNotify — payload-only view of an `Interface` values tree.
// ===========================================================================

/// Maps an [`Interface`] payload tree (`(bool, value)` leaves) onto its
/// values-only tree, dropping the notify flags. Lets closure operators
/// ([`Map`](super::Map) / [`Apply`](super::Apply)) keep ergonomic closure
/// signatures, e.g. `Fn((ArrayView<f64, 2>, ArrayView<f64, 2>))` for a two-port
/// input.
pub trait StripNotify: Interface {
    /// The values-only payload tree.
    type Plain<'a>: Copy;

    /// Drop the notify flags from a payload tree.
    fn plain<'a>(values: Self::Values<'a>) -> Self::Plain<'a>;
}

impl<V: ValueView> StripNotify for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = V::View<'a>;

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl<V: ValueView> StripNotify for RefViewPort<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Plain<'a> = &'a V::View<'a>;

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl<V: ValueView> StripNotify for RefViewPorts<V>
where
    for<'a> V::View<'a>: Sync,
{
    type Plain<'a> = &'a [&'a V::View<'a>];

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl StripNotify for () {
    type Plain<'a> = ();

    #[inline(always)]
    fn plain<'a>(_: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {}
}

macro_rules! impl_strip_notify_for_tuple {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: StripNotify,)+> StripNotify for ($($T,)+) {
            type Plain<'a> = ($($T::Plain<'a>,)+);

            #[inline(always)]
            fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
                ( $( <$T as StripNotify>::plain(values.$idx), )+ )
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
