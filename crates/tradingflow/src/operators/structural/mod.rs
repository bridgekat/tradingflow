//! Structural operators — everything that routes, gates, records or reshapes
//! the stream rather than computing over its values.
//!
//! * **Passthrough / conversion**: [`Where`], [`Cast`].
//! * **Gating**: [`Filter`] (whole-array cutoff) and [`Gate`] (the carry-safe
//!   view gate), plus the clock-driven [`Clocked`] / [`ResampleView`] /
//!   [`ResampleClocked`].
//! * **Recording**: [`Record`] appends an `Array` stream into a `Series`,
//!   stamping each row with event time — the bridge from the
//!   [`ArrayPort`](crate::ports::ArrayPort) to the
//!   [`SeriesPort`](crate::ports::SeriesPort) currency.
//! * **Reshape / combine**: [`Stack`] / [`StackSync`] (N → 1 along a **new**
//!   axis, `OUT == IN + 1`), [`Concat`] / [`ConcatSync`] (N → 1 along an
//!   **existing** axis, rank-preserving), and [`Split`] (1 → N row fan-out,
//!   `OUT == IN - 1`).
//!
//! In the view currency every multi-input combine takes `ArrayPorts<T, IN>`
//! (a contiguous slice of by-value strided views, wired straight from a slice
//! of independent [`ArrayPort`](crate::ports::ArrayPort) handles), so the old
//! owned/view operator split has collapsed into a single set of operators and
//! no value↔reference bridging exists anywhere. The combine into the output
//! cross-section is the irreducible panel→cross-section data movement (each
//! input materialized via `to_contiguous`); the per-stock selections upstream
//! are [`select`](super::transform::select)s.
//!
//! One operator per submodule; `reshape` holds the layout helpers the
//! stack/concat family share.

mod cast;
mod clocked;
mod concat;
mod concat_sync;
mod count;
mod filter;
mod gate;
mod keep_where;
mod last;
mod record;
mod resample_clocked;
mod resample_view;
mod reshape;
mod split;
mod stack;
mod stack_sync;

pub use cast::*;
pub use clocked::*;
pub use concat::*;
pub use concat_sync::*;
pub use count::*;
pub use filter::*;
pub use gate::*;
pub use keep_where::*;
pub use last::*;
pub use record::*;
pub use resample_clocked::*;
pub use resample_view::*;
// Only the shared state type escapes `reshape`; the copy/extent helpers stay
// private to the module.
pub use reshape::ReshapeState;
pub use split::*;
pub use stack::*;
pub use stack_sync::*;
