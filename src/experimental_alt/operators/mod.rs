//! Ported subset of built-in operators under the new wavefront
//! [`Operator`](super::operator::Operator) trait.
//!
//! Bodies are verbatim copies of `src/operators/*` modulo the trait
//! import path and the stronger `Output: Clone` bound (which every
//! existing Array/Series/unit/primitive output satisfies).
//!
//! Ported set (the minimum required to exercise every scheduler code
//! path via the acceptance tests):
//!
//! - [`Const`](r#const::Const)       — zero-input permanent-slot operator.
//! - [`Id`](id::Id)                  — single-input, zero-state passthrough.
//! - [`Add`](add::Add)               — two-input element-wise addition.
//! - [`Filter`](filter::Filter)      — predicate-gated propagation.
//! - [`Record`](record::Record)      — `Array<T> → Series<T>` append.
//! - [`Lag`](lag::Lag)               — stateful, N-steps-back lookup.
//! - [`RollingMean`](rolling::RollingMean) — count-window mean over a
//!   `Series<T>` input.
//! - [`Clocked`](clocked::Clocked)   — clock-gated wrapper around an
//!   inner operator.
//! - [`ConcatSync`](concat_sync::ConcatSync) — variadic slice input,
//!   selective copy driven by per-position produced bits.

pub mod add;
pub mod clocked;
pub mod concat_sync;
pub mod r#const;
pub mod filter;
pub mod id;
pub mod lag;
pub mod record;
pub mod rolling;

pub use add::Add;
pub use clocked::Clocked;
pub use concat_sync::ConcatSync;
pub use filter::Filter;
pub use id::Id;
pub use lag::Lag;
pub use record::Record;
pub use rolling::RollingMean;
pub use r#const::Const;
