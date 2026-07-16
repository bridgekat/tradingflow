//! Operator-contract helpers shared by TradingFlow operators: the [`ArrayViewMarker`]
//! view kind and the [`StripNotify`] payload helper.
//!
//! TradingFlow operators implement [`Operator`](crate::graph::Operator)
//! (notify-gated; the common case) or [`Segment`](crate::graph::Segment) (custom
//! gating, e.g. [`Clocked`](super::Clocked)) **directly** — there is no
//! TradingFlow-side operator trait or bridge, so operators compose with
//! the engine's combinators and the `segment!` fusion macro as-is.
//!
//! # The array currency
//!
//! Array-shaped edges all carry a borrowed, strided
//! [`ArrayView<'a, T, N>`](crate::ArrayView) by value through an
//! [`ArrayPort<T, N>`] leaf — one currency for owned-buffer outputs and
//! zero-copy slices alike. A compute operator homes its output buffer in `State`
//! (an owned [`Array<T, N>`](crate::Array)) and returns a view of it;
//! a slicing operator ([`SliceView`](super::SliceView)) re-derives a strided view
//! of its input by value, needing neither in-state shape nor an arena. Edges that
//! fan a single buffer into / out of `N` views ([`Split`](super::Split) outputs,
//! [`Stack`](super::Stack) inputs) use the [`ArrayPorts<T, N>`] group — the same
//! by-value views, N per wire, wired straight from/to plain [`ArrayPort`]
//! handles. Recorded-history edges carry a
//! [`SeriesView<'a, T, N>`](crate::SeriesView) window through [`SeriesPort`] /
//! [`SeriesPorts`] the same way.
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
//!   [`Stack`](super::Stack) / [`Concat`](super::Concat) depend on, and
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
//! # Event time
//!
//! Every operator here declares `type Context = Instant` — the engine's
//! graph-level context, which the [ingest](crate::ingest) driver sets to the
//! current batch's event time before each `stabilize`. So `compute` is *handed*
//! the timestamp; nothing is threaded through construction, and time never
//! becomes a graph dependency (a context write dirties no cone).
//!
//! Operators remain **pure** with respect to time: nearly all ignore the
//! argument, and only the few that genuinely stamp it read it (e.g.
//! [`Record`](super::Record)). Before the first batch the context holds
//! [`Instant::MIN`](crate::Instant::MIN), the floor set by `Scenario::new`; an
//! operator that must tell that build call apart uses the `init` flag, not the
//! timestamp.

use std::marker::PhantomData;

use crate::graph::{Interface, RefViewPort, RefViewPorts, ValueView, ViewPort, ViewPorts};

use crate::{ArrayView, Scalar, SeriesView};

// ===========================================================================
// View markers + port aliases — the engine-facing spelling of the currencies.
// ===========================================================================

/// [`ValueView`] kind passing a borrowed [`ArrayView<'a, T, N>`](crate::ArrayView)
/// across interfaces with the engine's per-generation lifetime — fully
/// borrow-checked zero-copy edges. Never spelled directly in operator
/// signatures: use the port aliases [`ArrayPort`] / [`ArrayPorts`]. The rank
/// `N` is compile-time; the operator's generic rank is inferred from the input
/// handle types at `push`.
pub struct ArrayViewMarker<T, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]` plus plain `usize`s, so it
// is covariant in `'a` — the only `ValueView` obligation.
unsafe impl<T: Scalar, const N: usize> ValueView for ArrayViewMarker<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
}

/// [`ValueView`] kind passing a borrowed
/// [`SeriesView<'a, T, N>`](crate::SeriesView) (a recorded-history window)
/// across interfaces — the [`ArrayViewMarker`] analogue for [`Series`](crate::Series)
/// edges. Never spelled directly: use [`SeriesPort`] / [`SeriesPorts`].
///
/// A [`SeriesView`] is the producer's *retained window* and its indices are
/// view-local (`0` is the oldest retained row, `len() - 1` the newest), so
/// consumers address history relative to the end of the window — which is
/// exactly what windowed operators do, and is retention-safe as long as the
/// producing [`Record`](super::Record)'s bound covers the consumer's look-back.
pub struct SeriesViewMarker<T, const N: usize>(PhantomData<T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]` plus a
// plain `Shape` — covariant in `'a`, the only `ValueView` obligation.
unsafe impl<T: Scalar, const N: usize> ValueView for SeriesViewMarker<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
}

/// A single port carrying a strided [`ArrayView<T, N>`](crate::ArrayView) by
/// value — the array-shaped edge currency.
pub type ArrayPort<T, const N: usize> = ViewPort<ArrayViewMarker<T, N>>;

/// A runtime-length group of [`ArrayPort`]s, payload `(&[bool],
/// &[ArrayView<T, N>])` — wires against a slice of independent [`ArrayPort`]
/// producer handles (`&[Handle<ArrayPort<T, N>>]`), no bridging adapters.
pub type ArrayPorts<T, const N: usize> = ViewPorts<ArrayViewMarker<T, N>>;

/// A single port carrying a [`SeriesView<T, N>`](crate::SeriesView) (recorded
/// history window) by value — the [`Series`](crate::Series) edge currency.
pub type SeriesPort<T, const N: usize> = ViewPort<SeriesViewMarker<T, N>>;

/// A runtime-length group of [`SeriesPort`]s, payload `(&[bool],
/// &[SeriesView<T, N>])`.
pub type SeriesPorts<T, const N: usize> = ViewPorts<SeriesViewMarker<T, N>>;

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

impl<V: ValueView> StripNotify for ViewPorts<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = &'a [V::View<'a>];

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
