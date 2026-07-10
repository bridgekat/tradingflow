//! 10-group layered backtest harness for single-factor evaluation (分层回测).
//!
//! For one factor on one universe, builds 10 decile portfolios plus an
//! equal-weight benchmark, each executed by a frictionless [`Benchmark`] trader,
//! and records each portfolio's NAV and turnover series. The driver then reduces
//! these to per-group annualized return / excess vs benchmark / excess
//! max-drawdown / average turnover, the monotonicity statistic, and the
//! top-minus-bottom long-short spread.
//!
//! Reuse only: decile membership is the `flowops` `rank_bucket` portfolio
//! (`RankBucket(low, high)` equal-weights the `[low, high)` percentile slice of
//! the factor within the universe; `RankBucket(0, 1)` is the equal-weight
//! benchmark), and execution is the native [`Benchmark`] trader (one-tick-delay
//! mark-on-close, dividend reinvest, 涨跌停 limit blocking). No new operators.

use flowgraph::typed::{Handle, RefPort};

use tradingflow::operators::{
    Clock, PyClassOperator, PyParams, as_view, benchmark, map, own, py_class_operator, record,
    turnover,
};
use tradingflow::{Array, ArrayView, Scenario, Series};

use super::AvH;

/// Number of layered-backtest groups (deciles).
pub const NUM_GROUPS: usize = 10;

/// Recorded NAV / turnover series for the decile portfolios and the benchmark.
pub struct DecileBacktest {
    /// Per-decile NAV series (group 0 = lowest factor value … group 9 = highest).
    pub decile_nav: Vec<Handle<RefPort<Series<f64, 0>>>>,
    /// Per-decile turnover series (L1 weight change per rebalance).
    pub decile_turnover: Vec<Handle<RefPort<Series<f64, 0>>>>,
    /// Equal-weight-universe benchmark NAV series.
    pub bench_nav: Handle<RefPort<Series<f64, 0>>>,
}

/// One `RankBucket(low, high)` portfolio → `Benchmark` trader → `(NAV record,
/// turnover record)`. `low == 0 && high == 1` is the equal-weight benchmark.
#[allow(clippy::too_many_arguments)]
fn bucket_nav(
    sc: &mut Scenario,
    universe: AvH,
    factor: AvH,
    close: AvH,
    adjusts: AvH,
    upper: AvH,
    lower: AvH,
    n: usize,
    clk: &Clock,
    low: f64,
    high: f64,
) -> (Handle<RefPort<Series<f64, 0>>>, Handle<RefPort<Series<f64, 0>>>) {
    // The Python portfolio consumes whole-array `RefPort` inputs; materialize the
    // two view inputs via `own`.
    let universe_ref = sc.push(own(), universe);
    let factor_ref = sc.push(own(), factor);
    let positions = sc.push(
        py_class_operator::<(RefPort<Array<f64, 1>>, RefPort<Array<f64, 1>>), 1>(
            "flowops.portfolios.mean.rank_bucket",
            PyParams::new()
                .int("num_stocks", n as i64)
                .float("low", low)
                .float("high", high),
            vec![n],
            clk,
        ),
        (universe_ref, factor_ref),
    );
    // The trader speaks the view currency; bridge the `RefPort` positions back.
    let positions_v = sc.push(as_view(), positions);
    let trader = sc.push(
        benchmark(n, 1.0, true),
        (positions_v, close, adjusts, upper, lower),
    );
    let nav = sc.push(
        map(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        trader,
    );
    let turnover = sc.push(turnover(), positions_v);
    (sc.push(record(clk), nav), sc.push(record(clk), turnover))
}

/// Build the 10-decile layered backtest for `factor` over `universe`.
#[allow(clippy::too_many_arguments)]
pub fn build_decile_backtest(
    sc: &mut Scenario,
    universe: AvH,
    factor: AvH,
    close: AvH,
    adjusts: AvH,
    upper: AvH,
    lower: AvH,
    n: usize,
    clk: &Clock,
) -> DecileBacktest {
    let mut decile_nav = Vec::with_capacity(NUM_GROUPS);
    let mut decile_turnover = Vec::with_capacity(NUM_GROUPS);
    for d in 0..NUM_GROUPS {
        let low = d as f64 / NUM_GROUPS as f64;
        let high = (d + 1) as f64 / NUM_GROUPS as f64;
        let (nav, turn) =
            bucket_nav(sc, universe, factor, close, adjusts, upper, lower, n, clk, low, high);
        decile_nav.push(nav);
        decile_turnover.push(turn);
    }
    let (bench_nav, _bench_turn) =
        bucket_nav(sc, universe, factor, close, adjusts, upper, lower, n, clk, 0.0, 1.0);
    DecileBacktest {
        decile_nav,
        decile_turnover,
        bench_nav,
    }
}
