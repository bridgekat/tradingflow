//! Trading simulation operators.
//!
//! A trader turns a strategy's target weights into a simulated portfolio NAV: it
//! reinvests dividends, executes rebalances against close prices, marks the book
//! to market, and reports `[holdings_value, cash]` (total NAV = their sum). This
//! is the step that closes the backtest loop, so having it in Rust lets a full
//! backtest run on the native operator set alone. [`Benchmark`], the
//! realistic-cost [`SimpleTrader`], and the stochastic [`RandomTrader`] are all
//! native Rust; there are no Python trader implementations.
//!
//! Each takes `((positions), (close), adjusts, upper_limit, lower_limit)`
//! over `[num_stocks]` arrays — positions and close as `(ClockPort,
//! ArrayPort<f64, 1>)` event streams (the positions clock marks a new target,
//! executed one tick delayed on the next close pulse), the rest as plain
//! arrays — and outputs `[2]` = `[holdings_value, cash]`.
//!
//! One operator per submodule; the private `core` submodule holds the port
//! aliases and the realistic-execution machinery shared by [`SimpleTrader`] and
//! [`RandomTrader`].

mod benchmark;
mod core;
mod random;
mod simple;

#[cfg(test)]
mod test_util;

pub use benchmark::*;
pub use random::*;
pub use simple::*;
