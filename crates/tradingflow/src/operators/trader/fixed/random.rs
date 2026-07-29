use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::base::{Exec, Fixed};

/// Execution policy for [`random`].
pub struct RandomExec {
    lot_size: f64,
    portfolio_size: usize,
    rng: StdRng,
}

impl RandomExec {
    pub fn new(lot_size: f64, portfolio_size: usize, seed: u64) -> Self {
        assert!(
            lot_size > 0.0,
            "trader::fixed::random: lot_size must be positive, got {lot_size}",
        );
        assert!(
            portfolio_size > 0,
            "trader::fixed::random: portfolio_size must be positive, got {portfolio_size}",
        );
        Self {
            lot_size,
            portfolio_size,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Exec for RandomExec {
    fn orders(
        &mut self,
        bids: &[f64],
        asks: &[f64],
        positions: &[f64],
        _cash: f64,
        target_values: &[f64],
    ) -> Vec<(usize, f64)> {
        let n = positions.len();

        // Efraimidis–Spirakis weighted reservoir: a candidate is a priced
        // stock with a positive target value, keyed by `u^(1/w)` with the
        // target value as its weight, and the `portfolio_size` largest keys
        // are the sample (probability proportional to weight, without
        // replacement). The keys are invariant under a common scaling of the
        // weights, so weighting by target *values* samples exactly as
        // weighting by target weights.
        let mut total = 0.0;
        let mut keyed: Vec<(f64, usize)> = Vec::new();
        for (i, ((&bid, &ask), &target_value)) in (bids.iter())
            .zip(asks.iter())
            .zip(target_values.iter())
            .enumerate()
        {
            let mean = (bid + ask) / 2.0;
            if mean.is_finite() && mean > 0.0 && target_value > 0.0 {
                total += target_value;
                let u: f64 = self.rng.random_range(0.0..1.0);
                let u = if u <= 0.0 { f64::MIN_POSITIVE } else { u };
                keyed.push((u.powf(1.0 / target_value), i));
            }
        }
        let mut res = Vec::new();
        if total <= 0.0 {
            return res;
        }
        keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        // Equal-value targets over the sample: each selected stock gets
        // `total / portfolio_size`, so an undersized candidate set invests
        // proportionally less rather than concentrating; every other priced
        // stock is targeted to zero.
        let mut hard = vec![0.0; n];
        for &(_, i) in keyed.iter().take(self.portfolio_size) {
            hard[i] = total / self.portfolio_size as f64;
        }
        for (i, ((&bid, &ask), &position)) in (bids.iter())
            .zip(asks.iter())
            .zip(positions.iter())
            .enumerate()
        {
            let mean = (bid + ask) / 2.0;
            if mean > 0.0 {
                let target_position = hard[i] / mean;
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

/// Stochastic trader that holds an equal-value random subset of
/// `portfolio_size` stocks, re-sampled per rebalance with probability
/// proportional to the target weights.
///
/// May introduce slight leverage due to rounding and fees.
///
/// Inputs and outputs: same as [`benchmark`](super::benchmark).
pub fn random(
    delayed: bool,
    initial_cash: f64,
    fee_base: f64,
    fee_rate: f64,
    lot_size: f64,
    portfolio_size: usize,
    seed: u64,
) -> Fixed<RandomExec> {
    Fixed::new(
        RandomExec::new(lot_size, portfolio_size, seed),
        delayed,
        initial_cash,
        fee_base,
        fee_rate,
    )
}
