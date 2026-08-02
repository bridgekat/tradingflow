//! Universe construction for the CICC 基本面因子手册 replication.
//!
//! Each universe is a cross-sectional `[num_stocks]` **mask**: `1.0` for
//! in-universe stocks, `NaN` otherwise. Masks are state wires recomputed from
//! the market cap **once per rebalance pulse** (a signalled
//! [`array_map_on`](tradingflow::operators::array::array_map_on)) and retained
//! in between, so a consumer reading between rebalances sees the
//! as-of-rebalance mask.
//!
//! Two flavours, both driven off circulating market cap (`close * circ_shares`):
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

use tradingflow::data::{Array, ArrayView, Instant};
use tradingflow::graph::Builder;
use tradingflow::operators::{array, elem, signal};
use tradingflow::ports::{ArrayPort, ArrayPortHandle, SignalPortHandle};
use tradingflow::time::UnixTime;

/// Full-market mask: `1.0` for stocks with finite positive market cap at the
/// last rebalance pulse, else `NaN`.
pub fn build_full_market_universe(
    sc: &mut Builder<Instant, UnixTime>,
    market_cap: ArrayPortHandle<f64, 1>,
    rebalance_signal: SignalPortHandle<0>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        signal::on_signal(array::array_map(|m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            Array::from_parts(
                [s.len()],
                s.iter()
                    .map(|&c| {
                        if c.is_finite() && c > 0.0 {
                            1.0
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
            )
        })),
        (rebalance_signal, market_cap),
    )
}

/// Cap-rank window mask: include stocks whose descending market-cap rank (0-based)
/// falls in `[lo, hi)`. Approximates a size index without real constituents.
pub fn build_caprank_universe(
    sc: &mut Builder<Instant, UnixTime>,
    market_cap: ArrayPortHandle<f64, 1>,
    rebalance_signal: SignalPortHandle<0>,
    lo: usize,
    hi: usize,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        signal::on_signal(array::array_map(move |m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            let n = s.len();
            let mut idx: Vec<usize> = (0..n).filter(|&i| s[i].is_finite() && s[i] > 0.0).collect();
            idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
            let mut mask: Box<[f64]> = vec![f64::NAN; n].into();
            for (rank, &i) in idx.iter().enumerate() {
                if rank >= lo && rank < hi {
                    mask[i] = 1.0;
                }
            }
            Array::from_parts([n], mask)
        })),
        (rebalance_signal, market_cap),
    )
}

/// AND a listing-age filter into a universe mask: keep a stock only if it is
/// in `universe` **and** its `aged` signal is finite. Passing the 244-trading-day
/// lag of the log price as `aged` excludes 次新 (stocks listed under ~1 year),
/// which the handbook drops — newly-listed A-shares have extreme first-year
/// returns that scramble fundamental-factor signals.
pub fn with_listing_filter(
    sc: &mut Builder<Instant, UnixTime>,
    universe: ArrayPortHandle<f64, 1>,
    aged: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        tradingflow::segment!(
            |u: ArrayPort<f64, 1>, aged: ArrayPort<f64, 1>| -> ArrayPort<f64, 1> {
                let zeros = array::zeros([1]) @ ();
                let keep = elem::and() @ (elem::gt() @ (u, zeros), elem::is_finite() @ aged);
                elem::indicator(1.0, f64::NAN) @ keep
            }
        ),
        (universe, aged),
    )
}

/// NaN out every entry where the universe mask is not `> 0`, leaving in-universe
/// values untouched. Applied **before** cross-sectional `Percentile` so the rank
/// is computed within the universe only. Multiplying by a `1.0 / NaN` indicator
/// leaves in-universe values bit-exact (`x * 1.0 == x`, including `±0` and `±∞`).
pub fn mask_to_universe(
    sc: &mut Builder<Instant, UnixTime>,
    data: ArrayPortHandle<f64, 1>,
    universe: ArrayPortHandle<f64, 1>,
) -> ArrayPortHandle<f64, 1> {
    sc.segment(
        array::binary_map(|&d: &f64, &u: &f64| if u > 0.0 { d } else { f64::NAN }),
        (data, universe),
    )
}

/// Market-cap-weighted index weights for the top-`k` stocks (proportional to
/// cap, normalised to 1; others zero).
pub fn calculate_index_weights(mc: &[f64], k: usize) -> Box<[f64]> {
    let n = mc.len();
    let mut w = vec![0.0f64; n].into();
    let mut valid: Box<[usize]> = (0..n)
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
/// recomputed from the circulating market cap once per rebalance pulse and
/// retained in between.
pub fn build_cap_weighted_universe(
    sc: &mut Builder<Instant, UnixTime>,
    market_cap: ArrayPortHandle<f64, 1>,
    rebalance_signal: SignalPortHandle<0>,
    index_size: usize,
) -> ArrayPortHandle<f64, 1> {
    let k = index_size;
    sc.segment(
        signal::on_signal(array::array_map(move |m: ArrayView<f64, 1>| {
            let s = m.to_contiguous();
            Array::from_parts([s.len()], calculate_index_weights(&s, k))
        })),
        (rebalance_signal, market_cap),
    )
}
