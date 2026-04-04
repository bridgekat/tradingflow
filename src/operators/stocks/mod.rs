//! Stock-specific operators.
//!
//! Operators in this module implement domain logic for equity market data
//! processing.
//!
//! - [`Annualize`] — convert year-to-date financial report values into
//!   annualised quarterly values using days-based scaling.
//! - [`ForwardAdjust`] — forward price adjustment for corporate actions
//!   (cash dividends and share dividends / bonus shares).

pub mod annualize;
pub mod forward_adjust;

pub use annualize::Annualize;
pub use forward_adjust::ForwardAdjust;
