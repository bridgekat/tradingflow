use super::base::{Exec, Fixed, FixedParams};

/// Operator parameters for [`simple`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleParams {
    pub delayed: bool,
    pub initial_cash: f64,
    pub fee_base_buy: f64,
    pub fee_base_sell: f64,
    pub fee_rate_buy: f64,
    pub fee_rate_sell: f64,
    pub lot_size: f64,
}

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
pub fn simple(params: SimpleParams) -> Fixed<SimpleExec> {
    Fixed::new(
        SimpleExec::new(params.lot_size),
        FixedParams {
            delayed: params.delayed,
            initial_cash: params.initial_cash,
            fee_base_buy: params.fee_base_buy,
            fee_base_sell: params.fee_base_sell,
            fee_rate_buy: params.fee_rate_buy,
            fee_rate_sell: params.fee_rate_sell,
        },
    )
}
