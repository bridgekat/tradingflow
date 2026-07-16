//! CICC 《量化基本面因子手册》 single-factor evaluation (RankIC) on A-shares.
//!
//! For each fundamental factor in the catalog ([`common::factors`]) this computes
//! the monthly cross-sectional **RankIC** vs the **next-period** return, and reports
//! the CICC 因子有效性检验 IC battery: **IC mean, IC_IR (= mean/std of the IC series),
//! IC t-stat (= mean/std·√n)**, with an optional 10-group layered backtest.
//!
//! The universe is the top-`--index-size` stocks by circulating market cap,
//! recomputed each rebalance — a **synthetic cap-ranked index**, uniform with the
//! other examples. It approximates the handbook's 全市场 (large `index_size`) /
//! 沪深300 / 中证500 sub-universes; the project has no point-in-time index
//! constituents, so this is a documented stand-in, NOT a true CSI constituent set.
//!
//! RankIC wiring: both the factor and the forward return are masked to the
//! universe, then cross-sectionally ranked ([`Percentile`]); `np.corrcoef` of two
//! rank vectors is Spearman. The forward return is `log adjusted close` resampled
//! onto the rebalance clock and first-differenced; [`InformationCoefficient`]
//! stores the factor at one rebalance and correlates it against the single return
//! tick that follows, so the emitted series is the next-period RankIC.
//!
//! Deviations (data limits): ST / 停牌 screens skipped (no data); index universes
//! approximated by the synthetic cap rank; monthly clock from calendar-day stepping.
//!
//! ```text
//! # whole market (index_size >= number of symbols):
//! cargo run --example factor_handbook --features python -- \
//!   --data-dir examples/data --factor all \
//!   --begin 2010-01-04 --end 2022-04-01 --index-size 6000 --rebalance-days 30
//! # large-cap synthetic index (≈ 沪深300): --index-size 300
//! ```

#[path = "common/mod.rs"]
mod common;

use tradingflow::graph::Operator;

use tradingflow::operators::{
    ArrayPort, lag_series, log, multiply, own, percentile, record, resample_clocked, stack,
};
use tradingflow::sources::pulse;
use tradingflow::{Array, ArrayView, Instant, Scenario, WallClock};

use clap::Parser;

/// CICC fundamental-factor handbook single-factor (RankIC) evaluation.
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    common: common::CommonArgs,
    /// Comma-separated factor names, or `all`.
    #[arg(long, default_value = "all")]
    factor: String,
    /// Factor catalog: `fundamental` (基本面因子手册) or `pv` (价量因子手册).
    #[arg(long, default_value = "fundamental")]
    catalog: String,
    /// Also run the 10-group layered backtest (分层回测) for the selected factors.
    #[arg(long, default_value_t = false)]
    backtest: bool,
    /// Also emit the time-averaged cross-sectional factor correlation matrix
    /// (`target/factor_handbook_corr.csv`) over the selected factors.
    #[arg(long, default_value_t = false)]
    correlations: bool,
}

/// Per-cross-section **pairwise-complete Pearson correlation** of the columns of
/// an `(N, K)` ranked-factor panel → a `(K, K)` matrix. Because the inputs are
/// the same percentile ranks the RankIC uses, this is the cross-sectional
/// **Spearman** correlation between every pair of factors at one rebalance;
/// pairwise-complete observations let factors with their own missing data still
/// contribute over the stocks they share. The driver records it on the rebalance
/// clock and averages over time to get the factor correlation matrix.
#[derive(Clone)]
struct CorrMatrix {
    k: usize,
}
struct CorrMatrixState {
    k: usize,
    out: Array<f64, 2>,
}
impl Operator for CorrMatrix {
    type Inputs = ArrayPort<f64, 2>;
    type Outputs = ArrayPort<f64, 2>;
    type Context = Instant;
    type State = CorrMatrixState;

    fn init(self) -> CorrMatrixState {
        let k = self.k;
        CorrMatrixState {
            k,
            out: Array::from_vec([k, k], vec![f64::NAN; k * k]),
        }
    }

    fn compute<'a, 'b: 'a>(
        (_, panel): (bool, ArrayView<'a, f64, 2>),
        _: &Instant,
        state: &'b mut CorrMatrixState,
        init: bool,
    ) -> (bool, ArrayView<'a, f64, 2>) {
        let k = state.k;
        if init {
            state.out = Array::from_vec([k, k], vec![f64::NAN; k * k]);
            return (false, state.out.view());
        }
        let n = panel.extents()[0]; // (N, K)
        let data = panel.to_contiguous(); // row-major: data[i*k + a]
        let data = &data[..];
        let out = state.out.as_mut_slice();
        for a in 0..k {
            for b in a..k {
                let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                let mut m = 0usize;
                for i in 0..n {
                    let (xa, xb) = (data[i * k + a], data[i * k + b]);
                    if xa.is_finite() && xb.is_finite() {
                        sx += xa;
                        sy += xb;
                        sxx += xa * xa;
                        syy += xb * xb;
                        sxy += xa * xb;
                        m += 1;
                    }
                }
                let r = if m >= 3 {
                    let mf = m as f64;
                    let cov = sxy - sx * sy / mf;
                    let vx = sxx - sx * sx / mf;
                    let vy = syy - sy * sy / mf;
                    if vx > 0.0 && vy > 0.0 {
                        cov / (vx.sqrt() * vy.sqrt())
                    } else {
                        f64::NAN
                    }
                } else {
                    f64::NAN
                };
                out[a * k + b] = r;
                out[b * k + a] = r;
            }
        }
        (true, state.out.view())
    }

    fn passthrough<'a, 'b: 'a>(
        _: (bool, ArrayView<'a, f64, 2>),
        _: &Instant,
        state: &'b CorrMatrixState,
    ) -> (bool, ArrayView<'a, f64, 2>) {
        (false, state.out.view())
    }
}

/// Annualized return of a recorded NAV series over `[begin, end]`, from the NAV
/// span and elapsed calendar time.
fn ann_return(session: &tradingflow::Session, h: NavHandle, begin: i64) -> f64 {
    let (ts, v) = common::read_scalar_series(session, h);
    let pts: Vec<(i64, f64)> = ts
        .into_iter()
        .zip(v)
        .filter(|(t, x)| *t >= begin && x.is_finite() && *x > 0.0)
        .collect();
    if pts.len() < 2 {
        return f64::NAN;
    }
    let (t0, v0) = *pts.first().unwrap();
    let (t1, v1) = *pts.last().unwrap();
    let years = (t1 - t0) as f64 / (365.0 * 86_400.0 * 1e9);
    if years <= 0.0 {
        return f64::NAN;
    }
    (v1 / v0).powf(1.0 / years) - 1.0
}

/// Mean of the finite values of a recorded series (used for average turnover).
fn mean_finite(session: &tradingflow::Session, h: NavHandle) -> f64 {
    let (_, v) = common::read_scalar_series(session, h);
    let f: Vec<f64> = v.into_iter().filter(|x| x.is_finite()).collect();
    if f.is_empty() {
        f64::NAN
    } else {
        f.iter().sum::<f64>() / f.len() as f64
    }
}

/// Monotonicity = |Spearman(group index, group annualized return)|: rank the
/// per-group returns and correlate with the group order. ~1 ⇒ strongly monotone.
fn monotonicity(anns: &[f64]) -> f64 {
    let finite: Vec<(usize, f64)> = anns
        .iter()
        .enumerate()
        .filter(|(_, a)| a.is_finite())
        .map(|(i, a)| (i, *a))
        .collect();
    let m = finite.len();
    if m < 2 {
        return f64::NAN;
    }
    // Rank the returns (ascending); the group index is already a rank.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| finite[i].1.partial_cmp(&finite[j].1).unwrap());
    let mut y = vec![0.0_f64; m];
    for (r, &i) in order.iter().enumerate() {
        y[i] = r as f64;
    }
    let x: Vec<f64> = finite.iter().map(|(i, _)| *i as f64).collect();
    let n = m as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for k in 0..m {
        sxy += (x[k] - mx) * (y[k] - my);
        sxx += (x[k] - mx).powi(2);
        syy += (y[k] - my).powi(2);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return f64::NAN;
    }
    (sxy / (sxx.sqrt() * syy.sqrt())).abs()
}

type NavHandle = tradingflow::graph::Handle<tradingflow::operators::SeriesPort<f64, 0>>;

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let symbols = common::load_symbols(&args.common.data_dir);
    let n = symbols.len();
    eprintln!(
        "loaded {n} symbols; universe = synthetic top-{} cap-ranked index \
         (a cap-weighted-index approximation, NOT a true 沪深300/中证500 \
         constituent set; pass --index-size >= {n} for the whole market)",
        args.common.index_size
    );

    let mut sc = Scenario::new(WallClock);

    let st = common::build_stacked(&mut sc, &symbols, &args.common);
    let catalog = match args.catalog.as_str() {
        "fundamental" => common::factors::build_factor_catalog(&mut sc, &st),
        "pv" => common::pv_factors::build_pv_catalog(&mut sc, &st),
        other => panic!("unknown catalog {other:?} (expected fundamental|pv)"),
    };

    let circ_market_cap = sc.push(multiply(), (st.close, st.circ_shares));
    let log_adj = sc.push(log(), st.adjusted_close);

    let rebalance_clock = sc.add_source(pulse(args.common.rebalance_instants()));

    // Universe = the top-`index_size` stocks by circulating market cap, recomputed
    // each rebalance — a synthetic cap-ranked index, uniform with the other
    // examples' `--index-size`. `index_size >= n` reduces to the whole market.
    // This is an APPROXIMATION of the handbook's 全市场 / 沪深300 / 中证500 sub-
    // universes (which need point-in-time index constituents we do not have).
    let universe = common::universe::build_caprank_universe(
        &mut sc,
        circ_market_cap,
        rebalance_clock,
        0,
        args.common.index_size,
    );

    // 次新 exclusion: require a finite adjusted price ~1 trading year (244 daily
    // ticks) ago, i.e. the stock was already listed a year before the rebalance.
    let log_adj_series = sc.push(record(), log_adj);
    let prior_year = sc.push(lag_series(244, f64::NAN), log_adj_series);
    let prior_year_reb = sc.push(resample_clocked(), (rebalance_clock, prior_year));
    let universe = common::universe::with_listing_filter(&mut sc, universe, prior_year_reb);

    // Forward (next-period) return, masked to the universe and cross-sectionally ranked.
    let fwd = common::factors::build_forward_return(&mut sc, log_adj, rebalance_clock);
    let fwd_masked = common::universe::mask_to_universe(&mut sc, fwd, universe);
    let fwd_rank = sc.push(percentile(), fwd_masked);
    // The Python IC metric consumes a whole-array `RefPort`; the forward rank is
    // shared across every factor, so bridge it once before the loop.
    let fwd_rank_ref = sc.push(own(), fwd_rank);

    // Daily ±10% price limits (涨跌停) for the trader's limit-blocking.
    let (upper, lower) = common::build_price_limits(&mut sc, st.close, 0.10);

    let want: Option<Vec<String>> = if args.factor == "all" {
        None
    } else {
        Some(
            args.factor
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
        )
    };

    let mut ic_handles = Vec::new();
    let mut ic_names = Vec::new();
    let mut ranks: Vec<common::AvH> = Vec::new();
    let mut backtests: Vec<(String, common::backtest::DecileBacktest)> = Vec::new();
    for (name, &feat) in catalog.names.iter().zip(catalog.feature.iter()) {
        if let Some(w) = &want
            && !w.iter().any(|x| x == name)
        {
            continue;
        }
        // Resample the (catalog-ranked+imputed) feature onto the rebalance clock,
        // mask to universe, and re-rank within it (monotonic for stocks that have
        // the factor; imputed stocks enter the RankIC at the median rank).
        let reb = sc.push(resample_clocked(), (rebalance_clock, feat));
        let masked = common::universe::mask_to_universe(&mut sc, reb, universe);
        let rank = sc.push(percentile(), masked);
        ranks.push(rank);
        // RankIC vs the forward return (corrcoef of two rank vectors = Spearman).
        ic_handles.push(common::ic::ic_series(&mut sc, rank, fwd_rank_ref, n));
        ic_names.push(name.clone());

        // 10-group layered backtest on the feature (RankBucket ranks internally).
        if args.backtest {
            let bt = common::backtest::build_decile_backtest(
                &mut sc, universe, feat, st.close, st.adjusts, upper, lower, n,
            );
            backtests.push((name.clone(), bt));
        }
    }

    // Optional cross-sectional factor correlation matrix: stack the K ranked
    // factor vectors into an `(N, K)` panel on the rebalance clock and reduce to a
    // `(K, K)` correlation matrix each rebalance (recorded; time-averaged below).
    let corr_handle = if args.correlations && ranks.len() >= 2 {
        let panel = sc.push(stack::<f64, 1, 2>(1), &ranks[..]); // (N, K)
        let corr = sc.push(CorrMatrix { k: ranks.len() }, panel); // (K, K)
        Some(sc.push(record(), corr))
    } else {
        None
    };

    let mut session = sc.build_with_threads(args.common.threads);
    let total = session.total_num_events();
    session
        .run(common::progress(total, args.common.begin()))
        .await;
    eprintln!();

    // IC summary per factor.
    println!(
        "\n{:>14}  {:>9}  {:>7}  {:>7}  {:>6}",
        "factor", "IC_mean", "IC_IR", "t", "n"
    );
    let mut cols: Vec<(String, Vec<i64>, Vec<f64>)> = Vec::new();
    for (name, h) in ic_names.iter().zip(ic_handles.iter()) {
        let (ts, v) = common::read_scalar_series(&session, *h);
        match common::ic::ic_stats(&v) {
            None => println!("{name:>14}  {:>9}", "n/a"),
            Some(s) => println!(
                "{name:>14}  {:>+9.4}  {:>+7.4}  {:>+7.2}  {:>6}",
                s.mean, s.ir, s.t, s.n
            ),
        }
        cols.push((name.clone(), ts, v));
    }

    let path = "target/factor_handbook_ic.csv";
    common::write_long_csv(path, &cols);
    println!("\nwrote {path}");

    // ---- Time-averaged factor correlation matrix ------------------------
    if let Some(h) = corr_handle {
        let s: tradingflow::SeriesView<f64, 2> = session.view(h);
        let k = ic_names.len();
        let stride = k * k;
        let nt = s.len();
        let mut sum = vec![0.0f64; stride];
        let mut cnt = vec![0usize; stride];
        for t in 0..nt {
            let row = s.at(t);
            for (c, &v) in row.iter().enumerate() {
                if v.is_finite() {
                    sum[c] += v;
                    cnt[c] += 1;
                }
            }
        }
        let mut csv = String::from("factor");
        for name in &ic_names {
            csv.push(',');
            csv.push_str(name);
        }
        csv.push('\n');
        for (a, name) in ic_names.iter().enumerate() {
            csv.push_str(name);
            for b in 0..k {
                let c = a * k + b;
                let v = if cnt[c] > 0 {
                    sum[c] / cnt[c] as f64
                } else {
                    f64::NAN
                };
                csv.push_str(&format!(",{v:.6}"));
            }
            csv.push('\n');
        }
        let corr_path = "target/factor_handbook_corr.csv";
        std::fs::write(corr_path, csv).unwrap_or_else(|e| panic!("write {corr_path}: {e}"));
        println!("wrote {corr_path}  ({k}×{k}, averaged over {nt} rebalances)");
    }

    // ---- Layered-backtest summary ---------------------------------------
    if args.backtest && !backtests.is_empty() {
        let begin = args.common.begin().as_nanos();
        println!("\n=== Layered backtest: 10 groups by ascending factor value ===");
        let mut nav_cols: Vec<(String, Vec<i64>, Vec<f64>)> = Vec::new();
        let mut decile_summary =
            String::from("factor,bench_ann,long_short,monotonicity,g1_turnover,g10_turnover");
        for g in 1..=10 {
            decile_summary.push_str(&format!(",g{g}_ann"));
        }
        decile_summary.push('\n');
        for (name, bt) in &backtests {
            let bench = ann_return(&session, bt.bench_nav, begin);
            let anns: Vec<f64> = bt
                .decile_nav
                .iter()
                .map(|&h| ann_return(&session, h, begin))
                .collect();
            let mono = monotonicity(&anns);
            let long_short = anns[anns.len() - 1] - anns[0];
            let g1_to = mean_finite(&session, bt.decile_turnover[0]);
            let g10_to = mean_finite(&session, bt.decile_turnover[bt.decile_turnover.len() - 1]);
            decile_summary.push_str(&format!(
                "{name},{bench:.6},{long_short:.6},{mono:.6},{g1_to:.6},{g10_to:.6}"
            ));
            for a in &anns {
                decile_summary.push_str(&format!(",{a:.6}"));
            }
            decile_summary.push('\n');
            println!(
                "\n{name}: bench_ann={:+.2}%  long-short(g10-g1)={:+.2}%  monotonicity={:.3}",
                bench * 100.0,
                long_short * 100.0,
                mono,
            );
            print!("  group ann%   : ");
            for a in &anns {
                print!("{:+6.1} ", a * 100.0);
            }
            print!("\n  excess vs bm%: ");
            for a in &anns {
                print!("{:+6.1} ", (a - bench) * 100.0);
            }
            println!(
                "\n  avg turnover : g1={:.2}  g10={:.2}",
                mean_finite(&session, bt.decile_turnover[0]),
                mean_finite(&session, bt.decile_turnover[bt.decile_turnover.len() - 1]),
            );
            for (d, &h) in bt.decile_nav.iter().enumerate() {
                let (ts, v) = common::read_scalar_series(&session, h);
                nav_cols.push((format!("{name}_g{}", d + 1), ts, v));
            }
            let (ts, v) = common::read_scalar_series(&session, bt.bench_nav);
            nav_cols.push((format!("{name}_bench"), ts, v));
        }
        let nav_path = "target/factor_handbook_nav.csv";
        common::write_long_csv(nav_path, &nav_cols);
        println!("\nwrote {nav_path}");
        let dec_path = "target/factor_handbook_decile.csv";
        std::fs::write(dec_path, decile_summary)
            .unwrap_or_else(|e| panic!("write {dec_path}: {e}"));
        println!("wrote {dec_path}");
    }
}
