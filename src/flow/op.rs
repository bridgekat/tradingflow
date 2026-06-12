//! Operator-contract helpers shared by the `flow` operators: the [`Clock`],
//! the [`StripNotify`] payload helper, and the [`ArrayValue`] view kind.
//!
//! TradingFlow operators implement [`flowgraph::typed::Operator`]
//! (notify-gated; the common case) or [`flowgraph::typed::Segment`] (custom
//! gating, e.g. [`Clocked`](super::Clocked)) **directly** — there is no
//! TradingFlow-side operator trait or bridge, so operators compose with
//! `flowgraph`'s combinators and the `segment!` fusion macro as-is.
//!
//! Conventions shared by every operator in this module tree:
//!
//! * **Output storage lives in `State`** (owned buffers); `compute` writes the
//!   state-owned buffer and returns `(notify, &out)` references into it — or,
//!   for view-emitting operators ([`Split`](super::Split)), lends
//!   [`ArraySlice`] views of its inputs through a per-generation
//!   [`Arena`](flowgraph::typed::Arena). State cells are boxed by the engine,
//!   so returned references stay valid across generations (`passthrough`
//!   never reallocates).
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

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use flowgraph::typed::{Interface, RefPort, RefViewPort, RefViewPorts, ValueView, ViewPort};

use crate::{Array, ArraySlice, Instant, Scalar};

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
// ArrayValue — the `ArraySlice` view kind for flowgraph ports.
// ===========================================================================

/// [`ValueView`] kind passing a borrowed [`ArraySlice`] across interfaces:
/// `ViewPort<ArrayValue<T>>` / `RefViewPort<ArrayValue<T>>` leaves carry
/// `ArraySlice<'a, T>` with the engine's per-generation lifetime — fully
/// borrow-checked zero-copy edges (see [`Split`](super::Split)).
pub struct ArrayValue<T>(PhantomData<T>);

// SAFETY: `ArraySlice<'a, T>` holds only `&'a` references (data + shape), so
// it is covariant in `'a` — the only `ValueView` obligation.
unsafe impl<T: Scalar> ValueView for ArrayValue<T> {
    type View<'a> = ArraySlice<'a, T>;
}

// ===========================================================================
// ArrayInput — array-payload leaves, abstracted over the edge kind.
// ===========================================================================

/// An interface leaf carrying array data, abstracted over **how** it rides the
/// wire: an owned [`Array`](crate::Array) by reference (`RefPort<Array<T>>`)
/// or a borrowed [`ArraySlice`] view (`ViewPort<ArrayValue<T>>` /
/// `RefViewPort<ArrayValue<T>>`). Array-shaped operators ([`Select`]
/// (super::Select), [`ForwardAdjust`](super::ForwardAdjust), [`Annualize`]
/// (super::Annualize)) are generic over this trait — one implementation serves
/// owned and zero-copy edges alike, with the owned kind as the default type
/// parameter so existing call sites are unchanged.
pub trait ArrayInput<T: Scalar>: Interface {
    /// The leaf's notify flag.
    fn notified(values: &Self::Values<'_>) -> bool;

    /// The array payload as a borrowed view.
    fn data<'a>(values: &Self::Values<'a>) -> ArraySlice<'a, T>;
}

impl<T: Scalar> ArrayInput<T> for RefPort<Array<T>> {
    #[inline(always)]
    fn notified(values: &Self::Values<'_>) -> bool {
        values.0
    }

    #[inline(always)]
    fn data<'a>(values: &Self::Values<'a>) -> ArraySlice<'a, T> {
        ArraySlice::from(values.1)
    }
}

impl<T: Scalar> ArrayInput<T> for RefViewPort<ArrayValue<T>> {
    #[inline(always)]
    fn notified(values: &Self::Values<'_>) -> bool {
        values.0
    }

    #[inline(always)]
    fn data<'a>(values: &Self::Values<'a>) -> ArraySlice<'a, T> {
        *values.1
    }
}

impl<T: Scalar> ArrayInput<T> for ViewPort<ArrayValue<T>> {
    #[inline(always)]
    fn notified(values: &Self::Values<'_>) -> bool {
        values.0
    }

    #[inline(always)]
    fn data<'a>(values: &Self::Values<'a>) -> ArraySlice<'a, T> {
        values.1
    }
}

// ===========================================================================
// StripNotify — payload-only view of an `Interface` values tree.
// ===========================================================================

/// Maps an [`Interface`] payload tree (`(bool, value)` leaves) onto its
/// values-only tree, dropping the notify flags. Lets closure operators
/// ([`Map`](super::Map) / [`Apply`](super::Apply)) keep their legacy closure
/// signatures, e.g. `Fn((&Array<f64>, &Array<f64>)) -> T` for a two-port input.
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
