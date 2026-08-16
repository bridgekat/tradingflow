use super::base::{Exec, Fixed, FixedParams};

/// Operator parameters for [`fractional`].
#[derive(Debug, Clone, Copy, Default)]
pub struct FractionalParams {
    pub delayed: bool,
    pub initial_cash: f64,
    pub fee_base_buy: f64,
    pub fee_base_sell: f64,
    pub fee_rate_buy: f64,
    pub fee_rate_sell: f64,
}

/// Execution policy for [`fractional`].
pub struct FractionalExec;

impl Exec for FractionalExec {
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

/// Idealized trader that executes orders at the best bid/ask prices.
///
/// May introduce slight leverage due to bid-ask spread and fees.
///
/// See [module-level docs](super) for inputs and outputs.
pub fn fractional(params: FractionalParams) -> Fixed<FractionalExec> {
    Fixed::new(
        FractionalExec,
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

/// Idealized trader that executes orders at the best bid/ask prices.
///
/// May introduce slight leverage due to bid-ask spread.
///
/// See [module-level docs](super) for inputs and outputs.
pub fn benchmark(delayed: bool, initial_cash: f64) -> Fixed<FractionalExec> {
    fractional(FractionalParams {
        delayed,
        initial_cash,
        ..FractionalParams::default()
    })
}
