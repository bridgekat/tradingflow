use super::base::{Exec, Fixed};

/// Execution policy for [`simple`].
pub struct SimpleExec {
    lot_size: f64,
}

impl SimpleExec {
    pub fn new(lot_size: f64) -> Self {
        assert!(
            lot_size > 0.0,
            "trader::fixed::simple: lot_size must be positive, got {lot_size}",
        );
        Self { lot_size }
    }
}

impl Exec for SimpleExec {
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
                let rounded_position = (target_position / self.lot_size).round() * self.lot_size;
                let position_delta = rounded_position - position;
                if position_delta != 0.0 {
                    res.push((i, position_delta));
                }
            }
        }
        res
    }
}

/// Simple trader that executes orders at the best bid/ask prices. Positions
/// are rounded to the nearest lot size at rebalance.
///
/// May introduce slight leverage due to rounding and fees.
///
/// See [module-level docs](super) for inputs and outputs.
pub fn simple(
    delayed: bool,
    initial_cash: f64,
    fee_base: f64,
    fee_rate: f64,
    lot_size: f64,
) -> Fixed<SimpleExec> {
    Fixed::new(
        SimpleExec::new(lot_size),
        delayed,
        initial_cash,
        fee_base,
        fee_rate,
    )
}
