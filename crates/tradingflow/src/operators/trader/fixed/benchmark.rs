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
/// See [module-level docs](super) for inputs and outputs.
pub fn benchmark(delayed: bool, initial_cash: f64) -> Fixed<BenchmarkExec> {
    Fixed::new(BenchmarkExec, delayed, initial_cash, 0.0, 0.0)
}
