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

use tradingflow::operators::{Apply, ArrayValue, Clocked, Map};
use tradingflow::{Array, ArrayView, Scenario, ViewPort};

use super::AvH;

/// Full-market mask: `1.0` for stocks with finite positive market cap this
/// rebalance, else `NaN`.
pub fn build_full_market_universe(
    sc: &mut Scenario,
    market_cap: AvH,
    rebalance_clock: Handle<RefPort<()>>,
) -> AvH {
    sc.add_operator(
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
    sc.add_operator(
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
    sc.add_operator(
        Apply::<(ViewPort<ArrayValue<f64, 1>>, ViewPort<ArrayValue<f64, 1>>), f64, 1, _>::new(
            |(u, a): (ArrayView<f64, 1>, ArrayView<f64, 1>)| {
                let (us, as_) = (u.to_contiguous(), a.to_contiguous());
                Array::from_vec(
                    [us.len()],
                    (0..us.len())
                        .map(|i| if us[i] > 0.0 && as_[i].is_finite() { 1.0 } else { f64::NAN })
                        .collect(),
                )
            },
        ),
        (universe, aged),
    )
}

/// NaN out every entry where the universe mask is not `> 0`, leaving in-universe
/// values untouched. Applied **before** cross-sectional `Percentile` so the rank
/// is computed within the universe only.
pub fn mask_to_universe(sc: &mut Scenario, data: AvH, universe: AvH) -> AvH {
    sc.add_operator(
        Apply::<(ViewPort<ArrayValue<f64, 1>>, ViewPort<ArrayValue<f64, 1>>), f64, 1, _>::new(
            |(f, u): (ArrayView<f64, 1>, ArrayView<f64, 1>)| {
                let (fs, us) = (f.to_contiguous(), u.to_contiguous());
                Array::from_vec(
                    [fs.len()],
                    (0..fs.len())
                        .map(|i| if us[i] > 0.0 { fs[i] } else { f64::NAN })
                        .collect(),
                )
            },
        ),
        (data, universe),
    )
}
