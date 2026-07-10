//! CLI arguments and the symbol universe file.

use std::fs;

use tradingflow::Instant;
use tradingflow::data::days_from_civil;

/// Parse a `YYYY-MM-DD` string into days since 1970-01-01 (a `clap` value parser,
/// so a malformed date yields a usage error rather than a panic).
pub fn parse_date_days(s: &str) -> Result<i64, String> {
    let parts: Vec<&str> = s.split('-').collect();
    let err = || format!("invalid date `{s}` (expected YYYY-MM-DD)");
    if parts.len() != 3 {
        return Err(err());
    }
    let y: i64 = parts[0].parse().map_err(|_| err())?;
    let m: i64 = parts[1].parse().map_err(|_| err())?;
    let d: i64 = parts[2].parse().map_err(|_| err())?;
    Ok(days_from_civil(y, m, d))
}

/// CLI args shared by every cross-sectional A-shares example. Embed it in an
/// example's own `#[derive(Parser)]` with `#[command(flatten)]`, so each example
/// declares exactly the extra args it needs (e.g. `--window` for the
/// feature-based ones, a `SYMBOL` positional for the single-stock plots) and
/// gets its own `--help`. All of these are **required** except `--sample-begin`
/// and `--threads` — there are no hidden "magic" defaults for the universe size,
/// rebalance cadence, or backtest window.
#[derive(clap::Args)]
pub struct CommonArgs {
    /// Data directory: the `examples/data` parquet tables + `symbol_list.csv`.
    #[arg(long)]
    pub data_dir: String,

    /// Backtest start date, e.g. 2022-01-01.
    #[arg(short = 'b', long = "begin", value_name = "DATE", value_parser = parse_date_days)]
    pub begin_days: i64,

    /// Backtest end date, e.g. 2024-12-31.
    #[arg(short = 'e', long = "end", value_name = "DATE", value_parser = parse_date_days)]
    pub end_days: i64,

    /// Number of stocks in the cap-weighted universe.
    #[arg(long)]
    pub index_size: usize,

    /// Rebalance every N calendar days.
    #[arg(long)]
    pub rebalance_days: i64,

    /// Warm-up sampling start; if omitted, defaults to `begin` − 400 days (enough
    /// to populate the 365-day TTM and the rolling feature windows).
    #[arg(long = "sample-begin", value_name = "DATE", value_parser = parse_date_days)]
    pub sample_begin: Option<i64>,

    /// Worker threads for the flowgraph `Pool` (0 = serial). `> 0` lets
    /// independent solve-bound operators (e.g. one cvxpy portfolio per
    /// risk-aversion) overlap via GIL release.
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

impl CommonArgs {
    pub fn begin(&self) -> Instant {
        Instant::from_utc_days(self.begin_days)
    }
    pub fn end(&self) -> Instant {
        Instant::from_utc_days(self.end_days)
    }
    pub fn data_start(&self) -> Instant {
        let days = match self.sample_begin {
            Some(s) => s.min(self.begin_days),
            None => self.begin_days - 400,
        };
        Instant::from_utc_days(days)
    }

    /// Rebalance trigger instants: every `rebalance_days` calendar days from
    /// `begin` through `end` inclusive (mirrors the Python `np.arange`).
    pub fn rebalance_instants(&self) -> Vec<Instant> {
        let mut out = Vec::new();
        let mut d = self.begin_days;
        while d <= self.end_days {
            out.push(Instant::from_utc_days(d));
            d += self.rebalance_days;
        }
        out
    }
}

/// Load all stock symbols from `<data_dir>/symbol_list.csv` (the `symbol`
/// column), in file order.
pub fn load_symbols(data_dir: &str) -> Vec<String> {
    let path = format!("{data_dir}/symbol_list.csv");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut lines = text.lines();
    let header = lines.next().expect("symbol_list header");
    let col = header
        .split(',')
        .position(|h| h.trim() == "symbol")
        .expect("`symbol` column");
    lines
        .filter_map(|l| l.split(',').nth(col).map(|s| s.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect()
}
