//! The realistic-cost, stochastic [`RandomTrader`] executor.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::data::{ArrayView, Instant, Layout};
use crate::graph::Segment;

use super::core::{TraderCore, TraderInputs, TraderValues, Vp, run_trader};

/// Random lot sizing ([`RandomTrader`]): Efraimidis–Spirakis weighted reservoir.
#[allow(clippy::too_many_arguments)]
fn random_lots(
    rng: &mut StdRng,
    portfolio_size: usize,
    current_value: f64,
    exec: &[f64],
    shares: &[f64],
    lot_size: f64,
    soft: &[f64],
    out: &mut [f64],
) {
    let n = out.len();
    out.iter_mut().for_each(|x| *x = 0.0);

    let mut weights = vec![0.0f64; n];
    let mut total = 0.0;
    for i in 0..n {
        if exec[i].is_finite() && exec[i] > 0.0 {
            let w = soft[i].max(0.0);
            weights[i] = w;
            total += w;
        }
    }
    if total <= 0.0 {
        return;
    }
    let n_candidates = weights.iter().filter(|&&w| w > 0.0).count();
    let n_select = portfolio_size.min(n_candidates);
    if n_select == 0 {
        return;
    }

    let mut keyed: Vec<(f64, usize)> = (0..n)
        .filter(|&i| weights[i] > 0.0)
        .map(|i| {
            let u: f64 = rng.random_range(0.0..1.0);
            let u = if u <= 0.0 { f64::MIN_POSITIVE } else { u };
            (u.powf(1.0 / weights[i]), i)
        })
        .collect();
    keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let hard = total / portfolio_size as f64;
    let mut hard_w = vec![0.0f64; n];
    for &(_, i) in keyed.iter().take(n_select) {
        hard_w[i] = hard;
    }
    for i in 0..n {
        let p = exec[i];
        if !p.is_finite() || p <= 0.0 {
            continue;
        }
        let target_shares = hard_w[i] * current_value / p;
        out[i] = ((target_shares - shares[i]) / lot_size).round();
    }
}

/// Realistic-cost executor with **random** sizing.
pub struct RandomTrader {
    num_stocks: usize,
    portfolio_size: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
    seed: u64,
}

impl RandomTrader {
    pub fn new(
        num_stocks: usize,
        portfolio_size: usize,
        initial_cash: f64,
        lot_size: f64,
        fee_base: f64,
        fee_rate: f64,
        seed: u64,
    ) -> Self {
        Self {
            num_stocks,
            portfolio_size,
            initial_cash,
            lot_size,
            fee_base,
            fee_rate,
            seed,
        }
    }
}

/// Runtime state for [`RandomTrader`]: the shared book plus the seeded RNG.
pub struct RandomTraderState {
    core: TraderCore,
    rng: StdRng,
    portfolio_size: usize,
}

impl Segment for RandomTrader {
    type Inputs = TraderInputs;
    type Outputs = Vp;
    type Context = Instant;
    type State = RandomTraderState;

    fn init(self, ((_, pos), ..): TraderValues<'_>) -> RandomTraderState {
        assert_eq!(
            pos.layout().len(),
            self.num_stocks,
            "trader: input length {} != num_stocks {}",
            pos.layout().len(),
            self.num_stocks,
        );
        RandomTraderState {
            core: TraderCore::new(
                self.num_stocks,
                self.initial_cash,
                self.lot_size,
                self.fee_base,
                self.fee_rate,
            ),
            rng: StdRng::seed_from_u64(self.seed),
            portfolio_size: self.portfolio_size,
        }
    }

    fn compute<'a, 'b: 'a>(
        values: TraderValues<'a>,
        state: &'b mut RandomTraderState,
        _: &Instant,
    ) -> ArrayView<'a, f64, 1> {
        let RandomTraderState {
            core,
            rng,
            portfolio_size,
        } = state;
        let ps = *portfolio_size;
        run_trader(core, values, |cv, exec, shares, ls, soft, out| {
            random_lots(rng, ps, cv, exec, shares, ls, soft, out)
        })
    }

    fn reset<'a, 'b: 'a>(
        _: TraderValues<'a>,
        state: &'b mut RandomTraderState,
    ) -> ArrayView<'a, f64, 1> {
        state.core.out.view()
    }
}

/// Stochastic executor: holds a seeded random `portfolio_size` subset.
pub fn random_trader(
    num_stocks: usize,
    portfolio_size: usize,
    initial_cash: f64,
    lot_size: f64,
    fee_base: f64,
    fee_rate: f64,
    seed: u64,
) -> RandomTrader {
    RandomTrader::new(
        num_stocks,
        portfolio_size,
        initial_cash,
        lot_size,
        fee_base,
        fee_rate,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::super::test_util::{arr, src};
    use super::*;
    use crate::graph::pool::Pool;
    use crate::graph::typed::Builder;

    #[test]
    fn random_trader_invests_and_is_seed_deterministic() {
        let nan = f64::NAN;
        let run = || {
            let mut b = Builder::new();
            let (pos, posv) = src(&mut b, &[nan; 5]);
            let (close, closev) = src(&mut b, &[nan; 5]);
            let (_adj, adjv) = src(&mut b, &[1.0; 5]);
            let (_up, upv) = src(&mut b, &[nan; 5]);
            let (_lo, lov) = src(&mut b, &[nan; 5]);
            let out = b.segment(
                RandomTrader::new(5, 2, 1000.0, 1.0, 0.0, 0.0, 0),
                (posv, closev, adjv.1, upv.1, lov.1),
            );
            let mut g = b.build();
            let mut pool = Pool::new(0);
            *g.state_mut(pos) = arr(&[0.2; 5]);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool, &Instant::MIN);
            *g.state_mut(close) = arr(&[10.0; 5]);
            g.stabilize(&mut pool, &Instant::MIN);
            let invested = g.view(out).as_slice().unwrap().to_vec();
            *g.state_mut(close) = arr(&[11.0, 9.0, 10.0, 10.0, 10.0]);
            g.stabilize(&mut pool, &Instant::MIN);
            let marked = g.view(out).as_slice().unwrap().to_vec();
            (invested, marked)
        };
        let (inv1, mk1) = run();
        let (_inv2, mk2) = run();

        assert_eq!(
            inv1,
            vec![1000.0, 0.0],
            "should be fully invested after rebalance"
        );
        assert_eq!(mk1, mk2, "same seed must give identical results");
        let nav = mk1[0] + mk1[1];
        assert!(
            nav > 900.0 && nav < 1100.0,
            "tick3 NAV {nav} out of held-leg bounds"
        );
    }
}
