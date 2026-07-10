//! Universe construction for the CICC 基本面因子手册 replication.
//!
//! Each universe is a cross-sectional `[num_stocks]` **mask** emitted on the
//! rebalance clock: `1.0` for in-universe stocks, `NaN` otherwise. Masks are
//! consumed by [`mask_to_universe`] (which NaNs out-of-universe entries before
//! cross-sectional ranking) and, in the layered backtest, as equal-weight
//! position vectors.
//!
//! Two flavours, both driven off circulating market cap (`close * circ_shares`)
//! sampled on the rebalance clock via [`Clocked`]:
//!
//! * [`build_full_market_universe`] — every stock with finite positive cap.
//! * [`build_caprank_universe`] — a descending-cap rank window `[lo, hi)`,
//!   approximating a size index (沪深300 ≈ `[0, 300)`, 中证500 ≈ `[300, 800)`).
//!   The project has no point-in-time index constituents, so this is a
//!   documented approximation.
//!
//! NOTE (data limitation): ST and 停牌 (suspension) screens are **skipped** — the
//! data set has neither flag. The 上市未满一年 (listing age) and 一字板
//! (limit-locked) screens are derivable and added in [`build_full_market_universe`]
//! once wired; for now the full-market universe is "any stock with valid cap".

use flowgraph::typed::{Handle, RefPort};

use tradingflow::operators::{
    and, greater_than, indicator, is_finite, multiply, ArrayValue, Clocked, Map,
};
use tradingflow::{Array, ArrayView, Scenario};

use super::{Av1, AvH};

/// Full-market mask: `1.0` for stocks with finite positive market cap this
/// rebalance, else `NaN`.
///
/// Clock-gated, so this stays a `Map` closure: [`Clocked`] wraps an `Operator`
/// (it needs `passthrough` to re-present the last mask between ticks), whereas a
/// fused `segment!` chain is a bare `Segment`.
pub fn build_full_market_universe(
    sc: &mut Scenario,
    market_cap: AvH,
    rebalance_clock: Handle<RefPort<()>>,
) -> AvH {
    sc.push(
        Clocked::new(Map::new(|m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            Array::from_vec(
                [s.len()],
                s.iter()
                    .map(|&c| if c.is_finite() && c > 0.0 { 1.0 } else { f64::NAN })
                    .collect(),
            )
        })),
        (rebalance_clock, market_cap),
    )
}

/// Cap-rank window mask: include stocks whose descending market-cap rank (0-based)
/// falls in `[lo, hi)`. Approximates a size index without real constituents.
pub fn build_caprank_universe(
    sc: &mut Scenario,
    market_cap: AvH,
    rebalance_clock: Handle<RefPort<()>>,
    lo: usize,
    hi: usize,
) -> AvH {
    sc.push(
        Clocked::new(Map::new(move |m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            let n = s.len();
            let mut idx: Vec<usize> =
                (0..n).filter(|&i| s[i].is_finite() && s[i] > 0.0).collect();
            idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
            let mut mask = vec![f64::NAN; n];
            for (rank, &i) in idx.iter().enumerate() {
                if rank >= lo && rank < hi {
                    mask[i] = 1.0;
                }
            }
            Array::from_vec([n], mask)
        })),
        (rebalance_clock, market_cap),
    )
}

/// AND a listing-age filter into a universe mask: keep a stock only if it is
/// in `universe` **and** its `aged` signal is finite. Passing the 244-trading-day
/// lag of the log price as `aged` excludes 次新 (stocks listed under ~1 year),
/// which the handbook drops — newly-listed A-shares have extreme first-year
/// returns that scramble fundamental-factor signals. Both inputs are on the
/// rebalance clock.
pub fn with_listing_filter(sc: &mut Scenario, universe: AvH, aged: AvH) -> AvH {
    sc.push(
        flowgraph::segment!(|u: Av1, aged: Av1| {
            let keep = and::<1>() @ (
                greater_than::<f64, 1>(0.0) @ u,
                is_finite::<f64, 1>() @ aged,
            );
            indicator::<f64, 1>(1.0, f64::NAN) @ keep
        }),
        (universe, aged),
    )
}

/// NaN out every entry where the universe mask is not `> 0`, leaving in-universe
/// values untouched. Applied **before** cross-sectional `Percentile` so the rank
/// is computed within the universe only. Multiplying by a `1.0 / NaN` indicator
/// leaves in-universe values bit-exact (`x * 1.0 == x`, including `±0` and `±∞`).
pub fn mask_to_universe(sc: &mut Scenario, data: AvH, universe: AvH) -> AvH {
    sc.push(
        flowgraph::segment!(|data: Av1, u: Av1| {
            let keep = indicator::<f64, 1>(1.0, f64::NAN) @ (greater_than::<f64, 1>(0.0) @ u);
            multiply::<f64, 1>() @ (data, keep)
        }),
        (data, universe),
    )
}

/// Market-cap-weighted index weights for the top-`k` stocks (proportional to
/// cap, normalised to 1; others zero).
pub fn calculate_index_weights(mc: &[f64], k: usize) -> Vec<f64> {
    let n = mc.len();
    let mut w = vec![0.0f64; n];
    let mut valid: Vec<usize> = (0..n)
        .filter(|&i| mc[i].is_finite() && mc[i] > 0.0)
        .collect();
    if valid.is_empty() {
        return w;
    }
    let k = k.min(valid.len());
    valid.sort_by(|&a, &b| mc[b].partial_cmp(&mc[a]).unwrap());
    let top = &valid[..k];
    let mut s = 0.0;
    for &i in top {
        w[i] = mc[i];
        s += mc[i];
    }
    if s > 0.0 {
        for &i in top {
            w[i] /= s;
        }
    }
    w
}

/// Cap-**weighted** top-`index_size` universe (a weight vector, not a mask),
/// recomputed on each rebalance tick. `market_cap` is the per-stock circulating
/// market cap; `rebalance_clock` is the `Handle<RefPort<()>>` of a
/// [`clock`](tradingflow::sources::clock) source.
pub fn build_cap_weighted_universe(
    sc: &mut Scenario,
    market_cap: AvH,
    rebalance_clock: Handle<RefPort<()>>,
    index_size: usize,
) -> AvH {
    let k = index_size;
    sc.push(
        Clocked::new(Map::new(move |m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            Array::from_vec([s.len()], calculate_index_weights(&s, k))
        })),
        (rebalance_clock, market_cap),
    )
}
