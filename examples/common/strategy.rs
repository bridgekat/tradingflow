//! The shared backtest spine: market data → universe → NAV, plus reporting.
//!
//! Every strategy example builds the same pipeline up to the predictor, and the
//! same one after the portfolio. [`Market`] is the front half (panel, features,
//! target, quotes, universe); [`Market::record_nav`] is the back half (positions
//! → trader → net value → record). What remains in each example is the part
//! that actually differs: which predictor, which optimizer, which trader.

use tradingflow::data::{Instant, Retention};
use tradingflow::graph::{Builder, Graph, Pool};
use tradingflow::operators::elem;
use tradingflow::operators::series::record_all;
use tradingflow::operators::trader::fixed::{Exec, Fixed, benchmark};
use tradingflow::ports::{ArrayPortHandle, SeriesPortHandle, SignalPortHandle};
use tradingflow::sources::basic::*;
use tradingflow::time::UnixTime;

use super::args::CommonArgs;
use super::data::{Stacked, build_stacked};
use super::features::{Features, build_features};
use super::models::Dims;
use super::target::{DELIST_DAYS, PRICE_LIMIT, TICK_SIZE, build_log_return_target, build_quotes};
use super::universe::build_cap_weighted_universe;

/// Starting capital for the reported NAV curves (the unit-cash traders are
/// scaled by this at report time).
pub const INITIAL_CASH: f64 = 1_000_000.0;
/// Trading days per year, for annualising.
pub const TRADING_DAYS: f64 = 252.0;

/// The market pipeline shared by every strategy example: the stacked panel, the
/// feature set, the log-return target, the synthetic quote book, and the
/// cap-weighted universe on the rebalance signal.
pub struct Market {
    pub st: Stacked,
    pub features: Features,
    /// The cap-weighted universe weights (read on the rebalance signal — also
    /// the benchmark weights, and what the Python operators consume).
    pub universe: ArrayPortHandle<f64, 1>,
    pub rebalance_signal: SignalPortHandle<0>,
    /// The daily close pulse.
    pub daily: SignalPortHandle<0>,
    /// The synthetic quote book on the daily pulse (see
    /// [`build_quotes`](super::target::build_quotes)) — what the traders
    /// execute and mark against.
    pub flags: ArrayPortHandle<bool, 1>,
    pub bids: ArrayPortHandle<f64, 1>,
    pub asks: ArrayPortHandle<f64, 1>,
    /// Log adjusted close — the source of returns (refreshed on `daily`).
    pub log_adj: ArrayPortHandle<f64, 1>,
    /// Raw winsorized log returns (the IC evaluators' target; refreshed on
    /// `daily`).
    pub target: ArrayPortHandle<f64, 1>,
    /// Raw winsorized log returns (the covariance predictor's target).
    pub target_series: SeriesPortHandle<f64, 1>,
    /// Cross-sectionally demeaned log returns (the mean predictor's target).
    pub demeaned_series: SeriesPortHandle<f64, 1>,
    /// Panel dimensions for the predictors.
    pub dims: Dims,
    /// Number of loaded symbols.
    pub n: usize,
}

impl Market {
    /// Build the spine. `retention` bounds the recorded feature/target panels —
    /// size it to the deepest consumer look-back.
    pub fn build(
        sc: &mut Builder<Instant, UnixTime>,
        symbols: &[String],
        args: &CommonArgs,
        retention: Retention,
    ) -> Self {
        let n = symbols.len();
        let st = build_stacked(sc, symbols, args);
        let features = build_features(sc, &st, retention);
        let circ_market_cap = sc.segment(elem::mul(), (st.close, st.circ_shares));
        let log_adj = sc.segment(elem::ln(), st.adjusted_close);
        let daily = st.daily;
        let (target, target_series, demeaned_series) =
            build_log_return_target(sc, log_adj, daily, retention);
        let (_, flags, bids, asks) =
            build_quotes(sc, daily, st.close, PRICE_LIMIT, TICK_SIZE, DELIST_DAYS);

        let rebalance_signal = sc.source(pulse(args.rebalance_instants()));
        let universe =
            build_cap_weighted_universe(sc, circ_market_cap, rebalance_signal, args.index_size);

        let dims = Dims {
            num_stocks: n,
            num_features: features.names.len(),
            universe_size: args.index_size,
            target_offset: 1,
        };
        Self {
            st,
            features,
            universe,
            rebalance_signal,
            daily,
            flags,
            bids,
            asks,
            log_adj,
            target,
            target_series,
            demeaned_series,
            dims,
            n,
        }
    }

    /// Trade a `(signal, weights)` position stream with `trader`, returning its
    /// net asset value. The [`Exec`] policy — frictionless
    /// [`benchmark`], lot-rounded `simple`, stochastic `random` — is the
    /// cost-model swap point; everything else (the quote book, the dividend
    /// stream, the schedule) is shared, so two traders on the same weights
    /// differ only in their execution.
    pub fn simulate<E: Exec>(
        &self,
        sc: &mut Builder<Instant, UnixTime>,
        trader: Fixed<E>,
        positions: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
    ) -> ArrayPortHandle<f64, 0> {
        let (_positions, _cash, net_value) = sc.segment(
            trader,
            (
                (self.daily, self.flags, self.bids, self.asks),
                (self.st.div_signals, self.st.share_divs, self.st.cash_divs),
                positions,
            ),
        );
        net_value
    }

    /// The cap-weighted index's NAV: trade the universe weights frictionlessly.
    pub fn index_nav(&self, sc: &mut Builder<Instant, UnixTime>) -> SeriesPortHandle<f64, 0> {
        self.record_nav(sc, (self.rebalance_signal, self.universe))
    }

    /// A `(signal, weights)` position stream's frictionless NAV: trade it with
    /// [`benchmark`] and record on the daily pulse.
    pub fn record_nav(
        &self,
        sc: &mut Builder<Instant, UnixTime>,
        positions: (SignalPortHandle<0>, ArrayPortHandle<f64, 1>),
    ) -> SeriesPortHandle<f64, 0> {
        let value = self.simulate(sc, benchmark(true, 1.0), positions);
        sc.segment(record_all(), (self.daily, value))
    }
}

/// Build, run to exhaustion with a progress bar, and return the finished session.
pub async fn run(sc: Builder<Instant, UnixTime>, args: &CommonArgs) -> Graph<Instant, UnixTime> {
    let mut session = sc.build();
    let mut pool = Pool::new(args.threads);
    let total = session.size_hint();
    session.run(&mut pool, super::progress(total)).await;
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
        session: &Graph<Instant, UnixTime>,
        label: impl Into<String>,
        begin_ns: i64,
        h: SeriesPortHandle<f64, 0>,
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
