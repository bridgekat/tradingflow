//! Operator-contract helpers shared by TradingFlow operators: the [`ArrayValue`]
//! / [`SeriesValue`] [`Value`] kinds (with the [`array_cell`] / [`series_cell`]
//! hand-poked cells over them), and the [`StripNotify`] payload helper.
//!
//! TradingFlow operators implement [`Operator`](crate::graph::Operator)
//! (notify-gated; the common case) or [`Segment`](crate::graph::Segment) (custom
//! gating, e.g. [`Clocked`](super::Clocked)) **directly** — there is no
//! TradingFlow-side operator trait or bridge, so operators compose with
//! the engine's combinators and the `segment!` fusion macro as-is.
//!
//! # The array currency
//!
//! **THE INVARIANT: every `Array`-shaped edge is an [`ArrayPort`] /
//! [`ArrayPorts`] edge and every `Series`-shaped edge is a [`SeriesPort`] /
//! [`SeriesPorts`] edge** — a borrowed view, passed by value. No graph edge
//! carries a whole `Array` / `Series` by reference (`RefPort<Array<…>>`), so
//! there are no bridge operators between an "owned" and a "view" spelling of
//! the same data: sources lend views of their cells ([`ViewSource`]), compute
//! operators lend views of their state, and whole-array consumers (the
//! Python host) accept views.
//!
//! Concretely, array-shaped edges all carry a borrowed, strided
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
//! Owned values **enter** the currency at source cells — the engine's own
//! sources, no TradingFlow-side segment: each marker's [`Value`] impl
//! nominates the view's owned form (`ArrayValue` is owned by an
//! [`Array`], `SeriesValue` by a [`Series`]), so a [`ViewSource`] cell
//! owns the value in node state (poke it via
//! [`state_mut`](crate::graph::Graph::state_mut), which dirties the node) and
//! borrows its view per generation, while whole-value payloads (a pulse's
//! `()`, a test's `i64`, an event batch `Vec<E>`) are
//! [`Ref<T>`](crate::graph::Ref) cells
//! ([`RefSource`](crate::graph::RefSource)). An
//! [`EventSource`](crate::ingest::EventSource) simply names its cell's kind
//! (`type Value`), and
//! [`Scenario::add_source`](crate::Scenario::add_source) allocates the
//! `ViewSource<S::Value>` directly — a source handle wires into the
//! operator library with no adapter and no owned-type dispatch trait;
//! hand-poked cells use [`array_cell`] / [`series_cell`] or the engine's
//! [`RefSource`](crate::graph::RefSource).
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

use crate::graph::{Interface, Value, ViewPort, ViewPorts, ViewSource};

use crate::{Array, ArrayView, Scalar, Series, SeriesView};

// ===========================================================================
// View markers + port aliases — the engine-facing spelling of the currencies.
// ===========================================================================

/// [`Value`] kind passing a borrowed [`ArrayView<'a, T, N>`](crate::ArrayView)
/// across interfaces with the engine's per-generation lifetime — fully
/// borrow-checked zero-copy edges. Never spelled directly in operator
/// signatures: use the port aliases [`ArrayPort`] / [`ArrayPorts`]. The rank
/// `N` is compile-time; the operator's generic rank is inferred from the input
/// handle types at `push`.
pub struct ArrayValue<T, const N: usize>(PhantomData<T>);

// SAFETY: `ArrayView<'a, T, N>` holds only `&'a [T]` plus plain `usize`s, so it
// is covariant in `'a` — the only `Value` obligation.
unsafe impl<T: Scalar, const N: usize> Value for ArrayValue<T, N> {
    type View<'a> = ArrayView<'a, T, N>;
    type Owned = Array<T, N>;

    #[inline(always)]
    fn borrow(owned: &Array<T, N>) -> ArrayView<'_, T, N> {
        owned.view()
    }
}

/// [`Value`] kind passing a borrowed
/// [`SeriesView<'a, T, N>`](crate::SeriesView) (a recorded-history window)
/// across interfaces — the [`ArrayValue`] analogue for [`Series`]
/// edges. Never spelled directly: use [`SeriesPort`] / [`SeriesPorts`].
///
/// A [`SeriesView`] is the producer's *retained window* and its indices are
/// view-local (`0` is the oldest retained row, `len() - 1` the newest), so
/// consumers address history relative to the end of the window — which is
/// exactly what windowed operators do, and is retention-safe as long as the
/// producing [`Record`](super::Record)'s bound covers the consumer's look-back.
pub struct SeriesValue<T, const N: usize>(PhantomData<T>);

// SAFETY: `SeriesView<'a, T, N>` holds only `&'a [Instant]` + `&'a [T]` plus a
// plain `Shape` — covariant in `'a`, the only `Value` obligation.
unsafe impl<T: Scalar, const N: usize> Value for SeriesValue<T, N> {
    type View<'a> = SeriesView<'a, T, N>;
    type Owned = Series<T, N>;

    #[inline(always)]
    fn borrow(owned: &Series<T, N>) -> SeriesView<'_, T, N> {
        owned.view()
    }
}

/// A single port carrying a strided [`ArrayView<T, N>`](crate::ArrayView) by
/// value — the array-shaped edge currency.
pub type ArrayPort<T, const N: usize> = ViewPort<ArrayValue<T, N>>;

/// A runtime-length group of [`ArrayPort`]s, payload `(&[bool],
/// &[ArrayView<T, N>])` — wires against a slice of independent [`ArrayPort`]
/// producer handles (`&[Handle<ArrayPort<T, N>>]`), no bridging adapters.
pub type ArrayPorts<T, const N: usize> = ViewPorts<ArrayValue<T, N>>;

/// A single port carrying a [`SeriesView<T, N>`](crate::SeriesView) (recorded
/// history window) by value — the [`Series`] edge currency.
pub type SeriesPort<T, const N: usize> = ViewPort<SeriesValue<T, N>>;

/// A runtime-length group of [`SeriesPort`]s, payload `(&[bool],
/// &[SeriesView<T, N>])`.
pub type SeriesPorts<T, const N: usize> = ViewPorts<SeriesValue<T, N>>;

// ===========================================================================
// Hand-poked cells — free constructors for ViewSource over the markers.
// ===========================================================================

/// A hand-poked [`Array`] cell: a [`ViewSource`] owning `initial` in node
/// state and lending its [`ArrayPort`] view (write it via
/// [`state_mut`](crate::graph::Graph::state_mut), which dirties the node).
/// This is what [`Scenario::add_source`](crate::Scenario::add_source)
/// allocates for an [`ArrayValue`] source; push one directly for a constant
/// or test cell. Whole-value cells use the engine's
/// [`RefSource`](crate::graph::RefSource) / [`Source`](crate::graph::Source).
pub fn array_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Array<T, N>,
) -> ViewSource<ArrayValue<T, N>, C> {
    ViewSource::new(initial)
}

/// A hand-poked [`Series`] cell lending its [`SeriesPort`] view — the
/// [`array_cell`] analogue for recorded-history cells.
pub fn series_cell<T: Scalar, const N: usize, C: Send + Sync + 'static>(
    initial: Series<T, N>,
) -> ViewSource<SeriesValue<T, N>, C> {
    ViewSource::new(initial)
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

impl<V: Value> StripNotify for ViewPort<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = V::View<'a>;

    #[inline(always)]
    fn plain<'a>(values: <Self as Interface>::Values<'a>) -> Self::Plain<'a> {
        values.1
    }
}

impl<V: Value> StripNotify for ViewPorts<V>
where
    for<'a> V::View<'a>: Copy + Send + Sync,
{
    type Plain<'a> = &'a [V::View<'a>];

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
