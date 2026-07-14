//! The shared backtest spine: market data → universe → NAV, plus reporting.
//!
//! Every strategy example builds the same pipeline up to the predictor, and the
//! same one after the portfolio. [`Market`] is the front half (panel, features,
//! target, limits, universe); [`Market::record_nav`] is the back half (positions
//! → trader → total value → record). What remains in each example is the part
//! that actually differs: which predictor, which optimizer, which trader.

use tradingflow::graph::{Handle, RefPort, ViewPort};

use tradingflow::operators::{ArrayValue, as_view, benchmark, log, map, multiply, own, record};
use tradingflow::sources::pulse;
use tradingflow::{Array, ArrayView, Instant, Retention, Scenario, Series, Session};

use super::AvH;
use super::args::CommonArgs;
use super::data::{Stacked, build_stacked};
use super::features::{FeatureSet, Features, build_strategy_features};
use super::models::Dims;
use super::target::{build_log_return_target, build_price_limits};
use super::universe::build_cap_weighted_universe;

/// Starting capital for the reported NAV curves (the unit-cash traders are
/// scaled by this at report time).
pub const INITIAL_CASH: f64 = 1_000_000.0;
/// Trading days per year, for annualising.
pub const TRADING_DAYS: f64 = 252.0;
/// Daily price-limit band (±10% for most A-shares).
pub const PRICE_LIMIT: f64 = 0.10;
/// Forward-return offset pairing `features[i]` with `target[i + 1]`.
pub const TARGET_OFFSET: i64 = 1;

/// A scalar view handle (a rank-0 `ArrayView` port).
pub type ScH = Handle<ViewPort<ArrayValue<f64, 0>>>;
/// A recorded scalar series handle — a NAV curve.
pub type NavH = Handle<RefPort<Series<f64, 0>>>;
/// A whole-array positions handle, as the Python portfolios emit them.
pub type PosH = Handle<RefPort<Array<f64, 1>>>;

/// Map a trader's `(holdings_value, cash)` view to its scalar total value.
pub fn total_value(sc: &mut Scenario, h: AvH) -> ScH {
    sc.push(
        map(|a: ArrayView<f64, 1>| Array::scalar(a.to_contiguous().iter().sum::<f64>())),
        h,
    )
}

/// The market pipeline shared by every strategy example: the stacked panel, the
/// feature set, the log-return target, the price limits, and the cap-weighted
/// universe on the rebalance clock.
pub struct Market {
    pub st: Stacked,
    pub features: Features,
    /// The cap-weighted universe as a view (also the benchmark weights).
    pub universe: AvH,
    /// The universe materialized once for the Python operators.
    pub universe_ref: PosH,
    pub rebalance_clock: Handle<RefPort<()>>,
    pub upper: AvH,
    pub lower: AvH,
    /// Log adjusted close — the source of returns.
    pub log_adj: AvH,
    /// Raw winsorized log returns, as a live view (the IC evaluators' target).
    pub target: AvH,
    /// Raw winsorized log returns (the covariance predictor's target).
    pub target_series: Handle<RefPort<Series<f64, 1>>>,
    /// Cross-sectionally demeaned log returns (the mean predictor's target).
    pub demeaned_series: Handle<RefPort<Series<f64, 1>>>,
    /// Panel dimensions for the predictors.
    pub dims: Dims,
    /// Number of loaded symbols.
    pub n: usize,
}

impl Market {
    /// Build the spine. `retention` bounds the recorded feature/target panels —
    /// size it to the deepest consumer look-back.
    pub fn build(
        sc: &mut Scenario,
        symbols: &[String],
        args: &CommonArgs,
        window: usize,
        set: FeatureSet,
        retention: Retention,
    ) -> Self {
        let n = symbols.len();
        let st = build_stacked(sc, symbols, args);
        let features = build_strategy_features(sc, &st, window, set, retention);
        let circ_market_cap = sc.push(multiply(), (st.close, st.circ_shares));
        let log_adj = sc.push(log(), st.adjusted_close);
        let (target, target_series, demeaned_series) =
            build_log_return_target(sc, log_adj, retention);
        let (upper, lower) = build_price_limits(sc, st.close, PRICE_LIMIT);

        let rebalance_clock = sc.add_source(pulse(args.rebalance_instants()));
        let universe =
            build_cap_weighted_universe(sc, circ_market_cap, rebalance_clock, args.index_size);
        // The Python predictor/portfolio operators consume whole-array
        // `RefPort`s; materialize the universe view once.
        let universe_ref = sc.push(own(), universe);

        let dims = Dims {
            num_stocks: n,
            num_features: features.names.len(),
            universe_size: args.index_size,
            target_offset: TARGET_OFFSET,
        };
        Self {
            st,
            features,
            universe,
            universe_ref,
            rebalance_clock,
            upper,
            lower,
            log_adj,
            target,
            target_series,
            demeaned_series,
            dims,
            n,
        }
    }

    /// Trade a position vector (already in the view currency) with `trader`,
    /// returning its scalar total value. `trader` is any of the native traders —
    /// `Benchmark` (frictionless), `SimpleTrader` / `RandomTrader` (lot + fee
    /// aware) — which is the cost-model swap point.
    pub fn simulate<T>(&self, sc: &mut Scenario, trader: T, positions: AvH) -> ScH
    where
        T: tradingflow::graph::Segment<
                Inputs = (
                    ViewPort<ArrayValue<f64, 1>>,
                    ViewPort<ArrayValue<f64, 1>>,
                    ViewPort<ArrayValue<f64, 1>>,
                    ViewPort<ArrayValue<f64, 1>>,
                    ViewPort<ArrayValue<f64, 1>>,
                ),
                Outputs = ViewPort<ArrayValue<f64, 1>>,
                Context = Instant,
            >,
    {
        let book = sc.push(
            trader,
            (
                positions,
                self.st.close,
                self.st.adjusts,
                self.upper,
                self.lower,
            ),
        );
        total_value(sc, book)
    }

    /// The cap-weighted index's NAV: trade the universe weights frictionlessly.
    pub fn index_nav(&self, sc: &mut Scenario) -> NavH {
        let value = self.simulate(sc, benchmark(self.n, 1.0, true), self.universe);
        sc.push(record(), value)
    }

    /// A Python portfolio's frictionless NAV: bridge its whole-array positions
    /// into the view currency, trade via `Benchmark`, sum, and record.
    pub fn record_nav(&self, sc: &mut Scenario, positions: PosH) -> NavH {
        let positions_v = sc.push(as_view(), positions);
        let value = self.simulate(sc, benchmark(self.n, 1.0, true), positions_v);
        sc.push(record(), value)
    }
}

/// Build, run to exhaustion with a progress bar, and return the finished session.
pub async fn run(sc: Scenario, args: &CommonArgs) -> Session {
    let mut session = sc.build_with_threads(args.threads);
    let total = session.total_num_events();
    session.run(super::progress(total, args.begin())).await;
    eprintln!(); // move past the finished bar line
    session
}

// ===========================================================================
// Reporting
// ===========================================================================

/// Drop samples before `begin_ns` and scale unit-cash NAVs to [`INITIAL_CASH`].
pub fn trim_scale(begin_ns: i64, ts: Vec<i64>, v: Vec<f64>) -> (Vec<i64>, Vec<f64>) {
    ts.into_iter()
        .zip(v)
        .filter(|(t, _)| *t >= begin_ns)
        .map(|(t, x)| (t, x * INITIAL_CASH))
        .unzip()
}

/// The last finite value of a NAV curve.
pub fn nav_final(v: &[f64]) -> f64 {
    v.iter()
        .rev()
        .copied()
        .find(|x| x.is_finite())
        .unwrap_or(f64::NAN)
}

/// Summary statistics of a daily NAV curve.
pub struct NavStats {
    pub cagr: f64,
    pub sharpe: f64,
    pub mdd: f64,
    /// The curve's last sample (NaN if empty) — not necessarily finite.
    pub final_value: f64,
    /// The curve's last *finite* sample — what to report when the tail of a
    /// curve can be NaN (e.g. a solver that failed on the final rebalance).
    pub final_finite: f64,
}

/// `(cagr, annualized Sharpe, max drawdown)` from a daily NAV series. The ratios
/// are NaN for curves with fewer than 10 finite positive samples.
pub fn nav_stats(v: &[f64]) -> NavStats {
    let final_value = v.last().copied().unwrap_or(f64::NAN);
    let final_finite = nav_final(v);
    let s: Vec<f64> = v
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if s.len() < 10 {
        return NavStats {
            cagr: f64::NAN,
            sharpe: f64::NAN,
            mdd: f64::NAN,
            final_value,
            final_finite,
        };
    }
    let years = s.len() as f64 / TRADING_DAYS;
    let cagr = (s[s.len() - 1] / s[0]).powf(1.0 / years) - 1.0;
    let rets: Vec<f64> = s.windows(2).map(|w| (w[1] / w[0]).ln()).collect();
    let mean = rets.iter().sum::<f64>() / rets.len() as f64;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
    let sd = var.sqrt();
    let sharpe = if sd > 0.0 {
        mean * TRADING_DAYS / (sd * TRADING_DAYS.sqrt())
    } else {
        f64::NAN
    };
    let mut peak = f64::MIN;
    let mut mdd = 0.0;
    for &x in &s {
        if x > peak {
            peak = x;
        }
        let dd = x / peak - 1.0;
        if dd < mdd {
            mdd = dd;
        }
    }
    NavStats {
        cagr,
        sharpe,
        mdd,
        final_value,
        final_finite,
    }
}

/// Accumulates the labelled NAV columns each strategy writes to CSV.
#[derive(Default)]
pub struct NavTable(pub Vec<(String, Vec<i64>, Vec<f64>)>);

impl NavTable {
    /// Read a recorded NAV, trim to the backtest window, scale to cash, and add
    /// it as a column — returning its statistics for printing.
    pub fn add(
        &mut self,
        session: &Session,
        label: impl Into<String>,
        begin_ns: i64,
        h: NavH,
    ) -> NavStats {
        let (t, v) = super::read_scalar_series(session, h);
        let (t, v) = trim_scale(begin_ns, t, v);
        let stats = nav_stats(&v);
        self.0.push((label.into(), t, v));
        stats
    }

    /// Write the accumulated columns and print the plot hint.
    pub fn write(&self, path: &str) {
        super::write_wide_csv(path, &self.0);
        println!("wrote {path}\nplot with:  python examples/plot_strategy.py {path}");
    }
}
