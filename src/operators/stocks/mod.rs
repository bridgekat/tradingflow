//! Stock-specific operators.
//!
//! Operators in this module implement domain logic for equity market data
//! processing.
//!
//! - [`ForwardAdjust`] — forward price adjustment for corporate actions
//!   (cash dividends and share dividends / bonus shares).

pub mod forward_adjust;

pub use forward_adjust::ForwardAdjust;
