//! Information-coefficient evaluation of the WorldQuant *101 Formulaic
//! Alphas* catalog, on the example data.

use chrono::{DateTime, NaiveDate};
use clap::Parser;
use indicatif::ProgressBar;
use std::collections::BTreeMap;
use tradingflow::{
    data::{Array, Axis, Duration, Instant, Schema},
    expr::Context,
    graph::{Builder, Pool},
    operators::{array, elem, feature, metric, rolling, series, signal},
    ports::SeriesPortHandle,
    sources::panel,
    time::UnixTime,
};

/// The symbols shipped in the example data, with a one-level industry tag
/// standing in for Alpha101's `IndClass.sector` / `.industry` /
/// `.subindustry`.
const SYMBOLS: [(&str, &str); 6] = [
    ("000001.SZ", "bank"),
    ("000002.SZ", "real_estate"),
    ("000858.SZ", "beverage"),
    ("600000.SH", "bank"),
    ("600519.SH", "beverage"),
    ("601398.SH", "bank"),
];

/// The `adv{d}` (mean daily dollar volume) windows registered as fields —
/// the set the 101 alphas reference.
const ADV_WINDOWS: [usize; 12] = [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180];

/// Cumulative daily IC curves of the Alpha101 catalog, on the example data.
#[derive(Parser)]
struct Args {
    /// Directory containing the data CSV files.
    #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/data"))]
    data_dir: String,
    /// Path of the cumulative IC curve CSV to write.
    #[arg(long, default_value = "features.csv")]
    output: String,
    /// First date (inclusive) to include in the curves and the summary,
    /// e.g. 2017-01-01. Alphas warm their windows up on all data before it.
    #[arg(long, value_parser = parse_date)]
    start: Option<Instant>,
    /// Last date to read (exclusive).
    #[arg(long, value_parser = parse_date)]
    end: Option<Instant>,
    /// Use the rank (Spearman) IC instead of the Pearson IC.
    #[arg(long)]
    rank: bool,
}

/// Parses a `YYYY-MM-DD` date into its midnight [`Instant`].
fn parse_date(s: &str) -> Result<Instant, String> {
    let date = NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|e| e.to_string())?;
    let ns = date
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .unwrap();
    Ok(Instant::from_offset(Duration::from_nanos(ns)))
}

/// Formats an [`Instant`] back into a `YYYY-MM-DD` date.
fn format_date(t: Instant) -> String {
    let dt = DateTime::from_timestamp_nanos(t.as_offset().as_nanos());
    dt.date_naive().format("%Y-%m-%d").to_string()
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dir = &args.data_dir;
    let n = SYMBOLS.len();

    // Create the thread pool.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());

    // Build the graph. Each table is a long-format CSV panel source over the
    // shared symbol axis.
    let mut b = Builder::new(UnixTime);
    let schema = Schema::new(SYMBOLS.map(|(symbol, _)| symbol));

    let (price_signals, prices) = b.source(
        panel::csv(
            format!("{dir}/prices.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec![
                "open".into(),
                "high".into(),
                "low".into(),
                "close".into(),
                "volume".into(),
                "amount".into(),
            ],
        )
        .with_time_range(None, args.end),
    );
    let (div_signals, divs) = b.source(
        panel::csv(
            format!("{dir}/dividends.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["share".into(), "cash".into()],
        )
        .with_time_range(None, args.end),
    );
    let (_equity_signals, equities) = b.source(
        panel::csv(
            format!("{dir}/equity_structures.csv"),
            "date",
            [("symbol".into(), Axis::Labeled(schema.clone()))],
            vec!["total".into()],
        )
        .with_time_range(None, args.end),
    );
    let (share_divs, cash_divs) = (divs[0], divs[1]);
    let total_shares = equities[0];

    // One scalar pulse per trading day (any symbol has a row).
    let daily = b.op(signal::any(), price_signals);

    // Daily price fields, converted to signaled-or-NaN representation:
    // that day's value, or NaN where the stock did not trade.
    let collect = |b: &mut Builder<Instant, UnixTime>, carried| {
        b.op(signal::collect(), (price_signals, carried, daily))
    };
    let open = collect(&mut b, prices[0]);
    let high = collect(&mut b, prices[1]);
    let low = collect(&mut b, prices[2]);
    let close = collect(&mut b, prices[3]);
    let volume = collect(&mut b, prices[4]);
    let amount = collect(&mut b, prices[5]);

    // Forward adjustment for dividends, whole cross-section at once. All
    // price-shaped fields are adjusted with the same multipliers; volume
    // divides.
    let (mult, adj_close) = b.op(
        feature::forward_adjust(),
        ((price_signals, close), (div_signals, share_divs, cash_divs)),
    );
    let adj_close = collect(&mut b, adj_close);
    let adj_open = b.op(elem::mul(), (open, mult));
    let adj_high = b.op(elem::mul(), (high, mult));
    let adj_low = b.op(elem::mul(), (low, mult));
    let adj_volume = b.op(elem::div(), (volume, mult));
    let raw_vwap = b.op(elem::div(), (amount, volume));
    let vwap = b.op(elem::mul(), (raw_vwap, mult));
    let returns = b.op(rolling::pct_change(1), (daily, adj_close));
    let cap = b.op(elem::mul(), (close, total_shares));

    // Industry tags, encoded as dense group ids in first-seen order.
    let mut ids = BTreeMap::new();
    let tags: Vec<u32> = SYMBOLS
        .iter()
        .map(|(_, tag)| {
            let next = ids.len() as u32;
            *ids.entry(*tag).or_insert(next)
        })
        .collect();
    let industry = b.val(array::constant(Array::from(tags)));

    // The factor-expression context, clocked by the daily pulse.
    let mut ctx = Context::new(daily, [n])
        .with_field("open", adj_open)
        .with_field("high", adj_high)
        .with_field("low", adj_low)
        .with_field("close", adj_close)
        .with_field("volume", adj_volume)
        .with_field("amount", amount)
        .with_field("vwap", vwap)
        .with_field("returns", returns)
        .with_field("cap", cap)
        .with_grouping("industry", industry);
    for d in ADV_WINDOWS {
        let adv = ctx
            .lower_str(&mut b, &format!("Mean($amount, {d})"))
            .expect("adv lowering cannot fail");
        ctx.add_field(format!("adv{d}"), adv);
    }

    // Lower every alpha, lag it by one trading day against the day's
    // realized return, and record the daily IC series.
    let recorded: Vec<(String, SeriesPortHandle<f64, 0>)> = ALPHA101
        .iter()
        .enumerate()
        .map(|(i, src)| {
            let name = format!("ALPHA{:03}", i + 1);
            let wire = ctx
                .lower_str(&mut b, src)
                .unwrap_or_else(|e| panic!("{name}: {e}\n  {src}"));
            let lagged = b.op(rolling::lag(1), (daily, wire));
            let ic = if args.rank {
                b.op(metric::feature::rank_ic(), (daily, lagged, returns))
            } else {
                b.op(metric::feature::ic(), (daily, lagged, returns))
            };
            (name, b.op(series::record_all(), (daily, ic)))
        })
        .collect();

    // Run the event loop until all sources are exhausted, with a progress bar
    // over the estimated total row count.
    let mut g = b.build();
    let bar = ProgressBar::new(g.size_hint().unwrap_or(0) as u64);
    g.run(&mut pool, |g, _| bar.set_position(g.num_events() as u64))
        .await;
    bar.finish();

    // Read the recorded daily IC series back out. All series are recorded on
    // the same daily pulse, so they share one timestamp axis.
    let instants: Vec<Instant> = g.view(recorded[0].1).instants().to_vec();
    let curves: Vec<Vec<f64>> = recorded
        .iter()
        .map(|&(ref name, handle)| {
            let view = g.view(handle);
            assert_eq!(view.instants(), &instants[..], "{name}: timestamp mismatch");
            view.to_contiguous().to_vec()
        })
        .collect();

    // Restrict to the requested date range.
    let first = match args.start {
        Some(start) => instants.partition_point(|&t| t < start),
        None => 0,
    };
    let instants = &instants[first..];
    assert!(!instants.is_empty(), "no trading days after --start");

    // Log the cumulative daily IC curves: one column per alpha, empty until
    // the alpha's first valid IC, carried over days with no valid IC.
    let mut csv = String::from("date");
    for (name, _) in &recorded {
        csv += &format!(",{name}");
    }
    csv += "\n";
    let mut sums = vec![(0.0, false); curves.len()];
    for (i, &t) in instants.iter().enumerate() {
        csv += &format_date(t);
        for (curve, (sum, seen)) in curves.iter().zip(sums.iter_mut()) {
            let ic = curve[first + i];
            if ic.is_finite() {
                *sum += ic;
                *seen = true;
            }
            csv += ",";
            if *seen {
                csv += &format!("{sum}");
            }
        }
        csv += "\n";
    }
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&args.output, csv).unwrap();
    println!(
        "wrote {} × {} cumulative {}IC points to {}",
        instants.len(),
        recorded.len(),
        if args.rank { "rank " } else { "" },
        args.output
    );

    // Print the summary: per-alpha mean IC and ICIR (mean / sample standard
    // deviation of the daily ICs).
    println!();
    println!("alpha      days    mean IC    IC std       ICIR");
    let mut stats = Vec::new();
    for ((name, _), curve) in recorded.iter().zip(curves.iter()) {
        let ics: Vec<f64> = curve[first..]
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .collect();
        let (mean, std) = if ics.len() >= 2 {
            let n = ics.len() as f64;
            let mean = ics.iter().sum::<f64>() / n;
            let var = ics.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var.sqrt())
        } else {
            (f64::NAN, f64::NAN)
        };
        let icir = mean / std;
        println!(
            "{name}  {:>5}  {mean:>+9.4}  {std:>8.4}  {icir:>+9.4}",
            ics.len()
        );
        stats.push((name.as_str(), ics.len(), mean, icir));
    }

    let valid: Vec<_> = stats.iter().filter(|s| s.3.is_finite()).collect();
    let mean_abs_ic = valid.iter().map(|s| s.2.abs()).sum::<f64>() / valid.len() as f64;
    let mean_abs_icir = valid.iter().map(|s| s.3.abs()).sum::<f64>() / valid.len() as f64;
    let best = valid
        .iter()
        .max_by(|a, b| a.3.abs().partial_cmp(&b.3.abs()).unwrap())
        .unwrap();
    println!();
    println!(
        "{} alphas over {} trading days ({}IC): mean |IC| {:.4}, mean |ICIR| {:.4}",
        valid.len(),
        instants.len(),
        if args.rank { "rank " } else { "" },
        mean_abs_ic,
        mean_abs_icir,
    );
    println!(
        "highest |ICIR|: {} (mean IC {:+.4}, ICIR {:+.4} over {} days)",
        best.0, best.2, best.3, best.1,
    );
}

/// The WorldQuant *101 Formulaic Alphas* (Kakushadze, 2016,
/// [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)), ported to the
/// expression vocabulary; `ALPHA101[i]` is Alpha#`i + 1`.
///
/// Porting rules (paper name → expression vocabulary): `rank` → `CSRank`,
/// `delay` → `Ref`, `correlation` → `Corr`, `covariance` → `Cov`, `delta` →
/// `Delta`, `decay_linear` → `WMA`, `ts_min`/`ts_max` → `Min`/`Max`,
/// `ts_argmin`/`ts_argmax` → `IdxMin`/`IdxMax`, `ts_rank` → `Rank`, `stddev`
/// → `Std`, pairwise `min`/`max` → `Less`/`Greater`, `x ? a : b` →
/// `If(x, a, b)`, `scale` → `Scale`, `indneutralize(x, IndClass.*)` →
/// `IndNeutralize(x, $industry)`, `signedpower(x, a)` →
/// `Sign(x) * Power(Abs(x), a)`.
///
/// Five decisions were forced during the port, each of which changes values
/// rather than merely spelling: fractional windows floor to tick counts;
/// boolean-valued alphas are wrapped as `If(a < b, 1, 0)`; `^` is `Power`;
/// all three industry levels become the one `$industry`; and `product` is
/// rewritten exactly (`product(x, 1)` is `x` in #29, `Log(product(x, w))` is
/// `Sum(Log(x), w)` in #81). The paper's genuine no-ops (`x^1` in #33, the
/// vwap identity in #59) are transcribed as written, so each expression can
/// be diffed against the paper; constant folding collapses them at lowering.
#[rustfmt::skip]
const ALPHA101: [&str; 101] = [
    // 1
    "CSRank(IdxMax(Sign(If($returns < 0, Std($returns, 20), $close)) \
     * Power(Abs(If($returns < 0, Std($returns, 20), $close)), 2), 5)) - 0.5",
    // 2
    "-1 * Corr(CSRank(Delta(Log($volume), 2)), CSRank(($close - $open) / $open), 6)",
    // 3
    "-1 * Corr(CSRank($open), CSRank($volume), 10)",
    // 4
    "-1 * Rank(CSRank($low), 9)",
    // 5
    "CSRank($open - Sum($vwap, 10) / 10) * (-1 * Abs(CSRank($close - $vwap)))",
    // 6
    "-1 * Corr($open, $volume, 10)",
    // 7
    "If($adv20 < $volume, (-1 * Rank(Abs(Delta($close, 7)), 60)) * Sign(Delta($close, 7)), -1)",
    // 8
    "-1 * CSRank(Sum($open, 5) * Sum($returns, 5) \
     - Ref(Sum($open, 5) * Sum($returns, 5), 10))",
    // 9
    "If(0 < Min(Delta($close, 1), 5), Delta($close, 1), \
     If(Max(Delta($close, 1), 5) < 0, Delta($close, 1), -1 * Delta($close, 1)))",
    // 10
    "CSRank(If(0 < Min(Delta($close, 1), 4), Delta($close, 1), \
     If(Max(Delta($close, 1), 4) < 0, Delta($close, 1), -1 * Delta($close, 1))))",
    // 11
    "(CSRank(Max($vwap - $close, 3)) + CSRank(Min($vwap - $close, 3))) * CSRank(Delta($volume, 3))",
    // 12
    "Sign(Delta($volume, 1)) * (-1 * Delta($close, 1))",
    // 13
    "-1 * CSRank(Cov(CSRank($close), CSRank($volume), 5))",
    // 14
    "(-1 * CSRank(Delta($returns, 3))) * Corr($open, $volume, 10)",
    // 15
    "-1 * Sum(CSRank(Corr(CSRank($high), CSRank($volume), 3)), 3)",
    // 16
    "-1 * CSRank(Cov(CSRank($high), CSRank($volume), 5))",
    // 17
    "((-1 * CSRank(Rank($close, 10))) * CSRank(Delta(Delta($close, 1), 1))) \
     * CSRank(Rank($volume / $adv20, 5))",
    // 18
    "-1 * CSRank(Std(Abs($close - $open), 5) + ($close - $open) + Corr($close, $open, 10))",
    // 19
    "(-1 * Sign(($close - Ref($close, 7)) + Delta($close, 7))) * (1 + CSRank(1 + Sum($returns, 250)))",
    // 20
    "((-1 * CSRank($open - Ref($high, 1))) * CSRank($open - Ref($close, 1))) \
     * CSRank($open - Ref($low, 1))",
    // 21
    "If(Sum($close, 8) / 8 + Std($close, 8) < Sum($close, 2) / 2, -1, \
     If(Sum($close, 2) / 2 < Sum($close, 8) / 8 - Std($close, 8), 1, \
     If(1 < $volume / $adv20 || $volume / $adv20 == 1, 1, -1)))",
    // 22
    "-1 * (Delta(Corr($high, $volume, 5), 5) * CSRank(Std($close, 20)))",
    // 23
    "If(Sum($high, 20) / 20 < $high, -1 * Delta($high, 2), 0)",
    // 24
    "If(Delta(Sum($close, 100) / 100, 100) / Ref($close, 100) < 0.05 \
     || Delta(Sum($close, 100) / 100, 100) / Ref($close, 100) == 0.05, \
     -1 * ($close - Min($close, 100)), -1 * Delta($close, 3))",
    // 25
    "CSRank(((-1 * $returns) * $adv20) * $vwap * ($high - $close))",
    // 26
    "-1 * Max(Corr(Rank($volume, 5), Rank($high, 5), 5), 3)",
    // 27
    "If(0.5 < CSRank(Sum(Corr(CSRank($volume), CSRank($vwap), 6), 2) / 2.0), -1, 1)",
    // 28
    "Scale(Corr($adv20, $low, 5) + ($high + $low) / 2 - $close)",
    // 29 — `product(x, 1)` is `x`; `min(x, 5)` is the pairwise minimum.
    "Less(CSRank(CSRank(Scale(Log(Sum(Min(CSRank(CSRank(-1 \
     * CSRank(Delta($close - 1, 5)))), 2), 1))))), 5) + Rank(Ref(-1 * $returns, 6), 5)",
    // 30
    "((1.0 - CSRank(Sign($close - Ref($close, 1)) + Sign(Ref($close, 1) - Ref($close, 2)) \
     + Sign(Ref($close, 2) - Ref($close, 3)))) * Sum($volume, 5)) / Sum($volume, 20)",
    // 31
    "CSRank(CSRank(CSRank(WMA(-1 * CSRank(CSRank(Delta($close, 10))), 10)))) \
     + CSRank(-1 * Delta($close, 3)) + Sign(Scale(Corr($adv20, $low, 12)))",
    // 32
    "Scale(Sum($close, 7) / 7 - $close) + 20 * Scale(Corr($vwap, Ref($close, 5), 230))",
    // 33
    "CSRank(-1 * Power(1 - $open / $close, 1))",
    // 34
    "CSRank((1 - CSRank(Std($returns, 2) / Std($returns, 5))) + (1 - CSRank(Delta($close, 1))))",
    // 35
    "(Rank($volume, 32) * (1 - Rank(($close + $high) - $low, 16))) * (1 - Rank($returns, 32))",
    // 36
    "2.21 * CSRank(Corr($close - $open, Ref($volume, 1), 15)) + 0.7 * CSRank($open - $close) \
     + 0.73 * CSRank(Rank(Ref(-1 * $returns, 6), 5)) + CSRank(Abs(Corr($vwap, $adv20, 6))) \
     + 0.6 * CSRank((Sum($close, 200) / 200 - $open) * ($close - $open))",
    // 37
    "CSRank(Corr(Ref($open - $close, 1), $close, 200)) + CSRank($open - $close)",
    // 38
    "(-1 * CSRank(Rank($close, 10))) * CSRank($close / $open)",
    // 39
    "(-1 * CSRank(Delta($close, 7) * (1 - CSRank(WMA($volume / $adv20, 9))))) \
     * (1 + CSRank(Sum($returns, 250)))",
    // 40
    "(-1 * CSRank(Std($high, 10))) * Corr($high, $volume, 10)",
    // 41
    "Power($high * $low, 0.5) - $vwap",
    // 42
    "CSRank($vwap - $close) / CSRank($vwap + $close)",
    // 43
    "Rank($volume / $adv20, 20) * Rank(-1 * Delta($close, 7), 8)",
    // 44
    "-1 * Corr($high, CSRank($volume), 5)",
    // 45
    "-1 * ((CSRank(Sum(Ref($close, 5), 20) / 20) * Corr($close, $volume, 2)) \
     * CSRank(Corr(Sum($close, 5), Sum($close, 20), 2)))",
    // 46
    "If(0.25 < (Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10, -1, \
     If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < 0, 1, \
     -1 * ($close - Ref($close, 1))))",
    // 47
    "((CSRank(1 / $close) * $volume) / $adv20) \
     * (($high * CSRank($high - $close)) / (Sum($high, 5) / 5)) - CSRank($vwap - Ref($vwap, 5))",
    // 48
    "IndNeutralize((Corr(Delta($close, 1), Delta(Ref($close, 1), 1), 250) * Delta($close, 1)) \
     / $close, $industry) / Sum(Power(Delta($close, 1) / Ref($close, 1), 2), 250)",
    // 49
    "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -1 * 0.1, \
     1, -1 * ($close - Ref($close, 1)))",
    // 50
    "-1 * Max(CSRank(Corr(CSRank($volume), CSRank($vwap), 5)), 5)",
    // 51
    "If((Ref($close, 20) - Ref($close, 10)) / 10 - (Ref($close, 10) - $close) / 10 < -1 * 0.05, \
     1, -1 * ($close - Ref($close, 1)))",
    // 52
    "(((-1 * Min($low, 5)) + Ref(Min($low, 5), 5)) \
     * CSRank((Sum($returns, 240) - Sum($returns, 20)) / 220)) * Rank($volume, 5)",
    // 53
    "-1 * Delta((($close - $low) - ($high - $close)) / ($close - $low), 9)",
    // 54
    "(-1 * (($low - $close) * Power($open, 5))) / (($low - $high) * Power($close, 5))",
    // 55
    "-1 * Corr(CSRank(($close - Min($low, 12)) / (Max($high, 12) - Min($low, 12))), \
     CSRank($volume), 6)",
    // 56
    "0 - 1 * (CSRank(Sum($returns, 10) / Sum(Sum($returns, 2), 3)) * CSRank($returns * $cap))",
    // 57
    "0 - 1 * (($close - $vwap) / WMA(CSRank(IdxMax($close, 30)), 2))",
    // 58
    "-1 * Rank(WMA(Corr(IndNeutralize($vwap, $industry), $volume, 3), 7), 5)",
    // 59
    "-1 * Rank(WMA(Corr(IndNeutralize($vwap * 0.728317 + $vwap * (1 - 0.728317), $industry), \
     $volume, 4), 16), 8)",
    // 60
    "0 - 1 * (2 * Scale(CSRank(((($close - $low) - ($high - $close)) / ($high - $low)) * $volume)) \
     - Scale(CSRank(IdxMax($close, 10))))",
    // 61
    "If(CSRank($vwap - Min($vwap, 16)) < CSRank(Corr($vwap, $adv180, 17)), 1, 0)",
    // 62
    "If(CSRank(Corr($vwap, Sum($adv20, 22), 9)) < CSRank(If(CSRank($open) + CSRank($open) \
     < CSRank(($high + $low) / 2) + CSRank($high), 1, 0)), 1, 0) * -1",
    // 63
    "(CSRank(WMA(Delta(IndNeutralize($close, $industry), 2), 8)) \
     - CSRank(WMA(Corr($vwap * 0.318108 + $open * (1 - 0.318108), Sum($adv180, 37), 13), 12))) * -1",
    // 64
    "If(CSRank(Corr(Sum($open * 0.178404 + $low * (1 - 0.178404), 12), Sum($adv120, 12), 16)) \
     < CSRank(Delta((($high + $low) / 2) * 0.178404 + $vwap * (1 - 0.178404), 3)), 1, 0) * -1",
    // 65
    "If(CSRank(Corr($open * 0.00817205 + $vwap * (1 - 0.00817205), Sum($adv60, 8), 6)) \
     < CSRank($open - Min($open, 13)), 1, 0) * -1",
    // 66
    "(CSRank(WMA(Delta($vwap, 3), 7)) + Rank(WMA(($low * 0.96633 + $low * (1 - 0.96633) - $vwap) \
     / ($open - ($high + $low) / 2), 11), 6)) * -1",
    // 67
    "Power(CSRank($high - Min($high, 2)), CSRank(Corr(IndNeutralize($vwap, $industry), \
     IndNeutralize($adv20, $industry), 6))) * -1",
    // 68
    "If(Rank(Corr(CSRank($high), CSRank($adv15), 8), 13) \
     < CSRank(Delta($close * 0.518371 + $low * (1 - 0.518371), 1)), 1, 0) * -1",
    // 69
    "Power(CSRank(Max(Delta(IndNeutralize($vwap, $industry), 2), 4)), \
     Rank(Corr($close * 0.490655 + $vwap * (1 - 0.490655), $adv20, 4), 9)) * -1",
    // 70
    "Power(CSRank(Delta($vwap, 1)), \
     Rank(Corr(IndNeutralize($close, $industry), $adv50, 17), 17)) * -1",
    // 71
    "Greater(Rank(WMA(Corr(Rank($close, 3), Rank($adv180, 12), 18), 4), 15), \
     Rank(WMA(Power(($low + $open) - ($vwap + $vwap), 2), 16), 4))",
    // 72
    "CSRank(WMA(Corr(($high + $low) / 2, $adv40, 8), 10)) \
     / CSRank(WMA(Corr(Rank($vwap, 3), Rank($volume, 18), 6), 2))",
    // 73
    "Greater(CSRank(WMA(Delta($vwap, 4), 2)), \
     Rank(WMA((Delta($open * 0.147155 + $low * (1 - 0.147155), 2) \
     / ($open * 0.147155 + $low * (1 - 0.147155))) * -1, 3), 16)) * -1",
    // 74
    "If(CSRank(Corr($close, Sum($adv30, 37), 15)) \
     < CSRank(Corr(CSRank($high * 0.0261661 + $vwap * (1 - 0.0261661)), CSRank($volume), 11)), \
     1, 0) * -1",
    // 75
    "If(CSRank(Corr($vwap, $volume, 4)) < CSRank(Corr(CSRank($low), CSRank($adv50), 12)), 1, 0)",
    // 76
    "Greater(CSRank(WMA(Delta($vwap, 1), 11)), \
     Rank(WMA(Rank(Corr(IndNeutralize($low, $industry), $adv81, 8), 19), 17), 19)) * -1",
    // 77
    "Less(CSRank(WMA((($high + $low) / 2 + $high) - ($vwap + $high), 20)), \
     CSRank(WMA(Corr(($high + $low) / 2, $adv40, 3), 5)))",
    // 78
    "Power(CSRank(Corr(Sum($low * 0.352233 + $vwap * (1 - 0.352233), 19), Sum($adv40, 19), 6)), \
     CSRank(Corr(CSRank($vwap), CSRank($volume), 5)))",
    // 79
    "If(CSRank(Delta(IndNeutralize($close * 0.60733 + $open * (1 - 0.60733), $industry), 1)) \
     < CSRank(Corr(Rank($vwap, 3), Rank($adv150, 9), 14)), 1, 0)",
    // 80
    "Power(CSRank(Sign(Delta(IndNeutralize($open * 0.868128 + $high * (1 - 0.868128), \
     $industry), 4))), Rank(Corr($high, $adv10, 5), 5)) * -1",
    // 81 — `Log(product(x, w))` is `Sum(Log(x), w)`.
    "If(CSRank(Sum(Log(CSRank(Power(CSRank(Corr($vwap, Sum($adv10, 49), 8)), 4))), 14)) \
     < CSRank(Corr(CSRank($vwap), CSRank($volume), 5)), 1, 0) * -1",
    // 82
    "Less(CSRank(WMA(Delta($open, 1), 14)), \
     Rank(WMA(Corr(IndNeutralize($volume, $industry), \
     $open * 0.634196 + $open * (1 - 0.634196), 17), 6), 13)) * -1",
    // 83
    "((CSRank(Ref(($high - $low) / (Sum($close, 5) / 5), 2)) * CSRank(CSRank($volume))) \
     / ((($high - $low) / (Sum($close, 5) / 5)) / ($vwap - $close)))",
    // 84
    "Sign(Rank($vwap - Max($vwap, 15), 20)) \
     * Power(Abs(Rank($vwap - Max($vwap, 15), 20)), Delta($close, 4))",
    // 85
    "Power(CSRank(Corr($high * 0.876703 + $close * (1 - 0.876703), $adv30, 9)), \
     CSRank(Corr(Rank(($high + $low) / 2, 3), Rank($volume, 10), 7)))",
    // 86
    "If(Rank(Corr($close, Sum($adv20, 14), 6), 20) \
     < CSRank(($open + $close) - ($vwap + $open)), 1, 0) * -1",
    // 87
    "Greater(CSRank(WMA(Delta($close * 0.369701 + $vwap * (1 - 0.369701), 1), 2)), \
     Rank(WMA(Abs(Corr(IndNeutralize($adv81, $industry), $close, 13)), 4), 14)) * -1",
    // 88
    "Less(CSRank(WMA((CSRank($open) + CSRank($low)) - (CSRank($high) + CSRank($close)), 8)), \
     Rank(WMA(Corr(Rank($close, 8), Rank($adv60, 20), 8), 6), 2))",
    // 89
    "Rank(WMA(Corr($low * 0.967285 + $low * (1 - 0.967285), $adv10, 6), 5), 3) \
     - Rank(WMA(Delta(IndNeutralize($vwap, $industry), 3), 10), 15)",
    // 90
    "Power(CSRank($close - Max($close, 4)), \
     Rank(Corr(IndNeutralize($adv40, $industry), $low, 5), 3)) * -1",
    // 91
    "(Rank(WMA(WMA(Corr(IndNeutralize($close, $industry), $volume, 9), 16), 3), 4) \
     - CSRank(WMA(Corr($vwap, $adv30, 4), 2))) * -1",
    // 92
    "Less(Rank(WMA(If(($high + $low) / 2 + $close < $low + $open, 1, 0), 14), 18), \
     Rank(WMA(Corr(CSRank($low), CSRank($adv30), 7), 6), 6))",
    // 93
    "Rank(WMA(Corr(IndNeutralize($vwap, $industry), $adv81, 17), 19), 7) \
     / CSRank(WMA(Delta($close * 0.524434 + $vwap * (1 - 0.524434), 2), 16))",
    // 94
    "Power(CSRank($vwap - Min($vwap, 11)), \
     Rank(Corr(Rank($vwap, 19), Rank($adv60, 4), 18), 2)) * -1",
    // 95
    "If(CSRank($open - Min($open, 12)) < Rank(Power(CSRank(Corr(Sum(($high + $low) / 2, 19), \
     Sum($adv40, 19), 12)), 5), 11), 1, 0)",
    // 96
    "Greater(Rank(WMA(Corr(CSRank($vwap), CSRank($volume), 3), 4), 8), \
     Rank(WMA(IdxMax(Corr(Rank($close, 7), Rank($adv60, 4), 3), 12), 14), 13)) * -1",
    // 97
    "(CSRank(WMA(Delta(IndNeutralize($low * 0.721001 + $vwap * (1 - 0.721001), $industry), 3), 20)) \
     - Rank(WMA(Rank(Corr(Rank($low, 7), Rank($adv60, 17), 4), 18), 15), 6)) * -1",
    // 98
    "CSRank(WMA(Corr($vwap, Sum($adv5, 26), 4), 7)) \
     - CSRank(WMA(Rank(IdxMin(Corr(CSRank($open), CSRank($adv15), 20), 8), 6), 8))",
    // 99
    "If(CSRank(Corr(Sum(($high + $low) / 2, 19), Sum($adv60, 19), 8)) \
     < CSRank(Corr($low, $volume, 6)), 1, 0) * -1",
    // 100
    "0 - 1 * ((1.5 * Scale(IndNeutralize(IndNeutralize(CSRank(((($close - $low) \
     - ($high - $close)) / ($high - $low)) * $volume), $industry), $industry)) \
     - Scale(IndNeutralize(Corr($close, CSRank($adv20), 5) \
     - CSRank(IdxMin($close, 30)), $industry))) * ($volume / $adv20))",
    // 101
    "($close - $open) / (($high - $low) + 0.001)",
];
