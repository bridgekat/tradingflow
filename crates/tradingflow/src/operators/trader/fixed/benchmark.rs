use super::base::{Exec, Fixed};

/// Execution policy for [`benchmark`].
pub struct BenchmarkExec;

impl Exec for BenchmarkExec {
    fn orders(
        &mut self,
        bids: &[f64],
        asks: &[f64],
        positions: &[f64],
        _cash: f64,
        target_values: &[f64],
    ) -> Vec<(usize, f64)> {
        let mut res = Vec::new();
        for (i, (((&bid, &ask), &position), &target_value)) in (bids.iter())
            .zip(asks.iter())
            .zip(positions.iter())
            .zip(target_values.iter())
            .enumerate()
        {
            let mean = (bid + ask) / 2.0;
            if mean > 0.0 {
                let target_position = target_value / mean;
                let position_delta = target_position - position;
                if position_delta != 0.0 {
                    res.push((i, position_delta));
                }
            }
        }
        res
    }
}

/// Frictionless and fractional trader that executes orders at the best bid/ask
/// prices.
///
/// May introduce slight leverage due to bid-ask spread and fees.
///
/// Inputs:
///
/// - `(price_signal, flags, bids, asks)`: the trading period signal,
///   with per-instrument inclusion flags (set to `false` to exclude instrument
///   from net asset value) and the best bid/ask prices at certain time point
///   during the trading period. Bid prices may be `-inf` and ask prices may be
///   `+inf` to indicate that the instrument cannot be sold or bought.
/// - `(div_signals, share_divs, cash_divs)`: per-instrument dividend event
///   signal, with the number of share dividends and cash dividends per share.
/// - `(rebalance_signal, target_weights)`: rebalance signal and the target
///   weights for each instrument (should sum to range `[0, 1]` if unleveraged).
///
/// Outputs:
///
/// - `positions`: the number of shares held for each instrument.
/// - `cash`: the amount of cash held.
/// - `net_value`: the total net asset value of the portfolio.
///
/// If some `rebalance_signal` coincides with `price_signal`, the trader
/// rebalances at the next `price_signal` if `delayed` is set.
///
/// Net asset value is computed against mark prices, which are the most recent
/// valid bid prices (with fallback to most recent ask prices, so limit-down
/// instruments still get a mark price update). If an asset's inclusion flag
/// is set to `false`, its mark price will be set to 0, until the flag is set
/// to `true` with a valid bid/ask price later.
pub fn benchmark(delayed: bool, initial_cash: f64) -> Fixed<BenchmarkExec> {
    Fixed::new(BenchmarkExec, delayed, initial_cash, 0.0, 0.0)
}
