//! Operators that simulate trading at a fixed price (e.g. closing price)
//! per period. Prices are given as [`f64`] to ensure precision.
//!
//! # Inputs
//!
//! - `(price_signal, flags, bids, asks)`: the trading period signal,
//!   with per-instrument inclusion flags (set to `false` to exclude instrument
//!   from net asset value) and the best bid/ask prices at certain time point
//!   during the trading period. Bid prices may be `-inf` and ask prices may be
//!   `+inf` to indicate that the instrument cannot be sold or bought.
//! - `(div_signals, share_divs, cash_divs)`: per-instrument dividend event
//!   signal, with the number of share dividends and cash dividends per share.
//! - `(rebalance_signal, target_weights)`: rebalance signal and the target
//!   weights for each instrument (should sum to range `[0, 1]` if unleveraged).
//!
//! # Outputs
//!
//! - `positions`: the number of shares held for each instrument.
//! - `cash`: the amount of cash held.
//! - `net_value`: the total net asset value of the portfolio.
//!
//! If some `rebalance_signal` coincides with `price_signal`, the trader
//! rebalances at the next `price_signal` if `delayed` is set.
//!
//! Net asset value is computed against mark prices, which are the most recent
//! valid bid prices (with fallback to most recent ask prices, so limit-down
//! instruments still get a mark price update). If an asset's inclusion flag
//! is set to `false`, its mark price will be set to 0, until the flag is set
//! to `true` with a valid bid/ask price later.

mod base;
mod benchmark;
mod random;
mod simple;

pub use base::{Exec, Fixed, FixedParams, FixedState};
pub use benchmark::benchmark;
pub use random::{RandomParams, random};
pub use simple::{SimpleParams, simple};
