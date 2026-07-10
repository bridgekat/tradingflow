//! Shared scaffolding for the A-shares cross-sectional examples.
//!
//! Pulled in by each strategy/plot example via `#[path = "common/mod.rs"] mod
//! common;`. It uses no Python operators (builds with the default feature set):
//! it builds the data pipeline (columnar parquet panel + financial-report
//! sources, the stacked cross-sectional panel, the feature sets, the
//! cap-weighted universe, the log-return target, and price limits) entirely from
//! native operators. Examples add their `flowops` predictor/portfolio operators
//! (and any cvxpy ones) on top.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`args`] | CLI args (`CommonArgs`), date parsing, the symbol list |
//! | [`data`] | parquet sources → the stacked `[num_stocks]` panel (`Stacked`) |
//! | [`factors`] / [`pv_factors`] | the CICC fundamental / price-volume factor catalogs |
//! | [`features`] | feature panels (`Features`, `FeatureSet`) over those catalogs |
//! | [`universe`] | universe masks and the cap-weighted index |
//! | [`target`] | the log-return prediction target and price limits |
//! | [`backtest`] | the decile-backtest harness (needs the `python` feature) |
//! | [`report`] | progress bar and CSV output |
//!
//! Every item is re-exported here, so examples keep using `common::*` paths.
//!
//! Differences from the Python original, all intentional:
//!
//! * Calendar/timezone handling is the caller's job in Rust (the core has no
//!   tz database), so rebalance dates are generated from a plain
//!   `Duration::from_days` step, and date strings are turned into `Instant`s
//!   via [`Instant::from_utc_days`](tradingflow::Instant::from_utc_days) — a
//!   UTC-midnight → TAI parse matching `ParquetPanelSource`'s.
//! * The Python lambdas (`calculate_index_weights`, cross-sectional demean,
//!   price-limit rounding) become native Rust closures fed to `Map`/`Apply`.
//! * All symbols in `symbol_list.csv` are loaded (matching the original);
//!   selection of the top-`index_size` constituents happens in-graph.

#![allow(dead_code)] // not every example uses every helper
#![allow(unused_imports)] // the re-exports below serve all examples, not each one

use flowgraph::typed::{Handle, ViewPort};

use tradingflow::operators::ArrayValue;

// These modules wrap `flowops` operators (`PyClassOperator`), so they only
// compile with the `python` feature. Their consumers (the strategy examples,
// `factor_handbook`) are themselves `required-features = ["python"]`; gating
// keeps the native examples — which pull in this `common` module too —
// buildable without Python.
#[cfg(feature = "python")]
pub mod backtest;
#[cfg(feature = "python")]
pub mod ic;
#[cfg(feature = "python")]
pub mod models;
#[cfg(feature = "python")]
pub mod strategy;

pub mod args;
pub mod data;
pub mod factors;
pub mod features;
pub mod pv_factors;
pub mod report;
pub mod target;
pub mod universe;

pub use args::{CommonArgs, load_symbols, parse_date_days};
pub use data::{Stacked, build_stacked};
pub use features::{
    FeatureSet, Features, build_factor_features, build_features, build_strategy_features,
};
pub use report::{date_str, progress, read_scalar_series, write_long_csv, write_wide_csv};
pub use target::{build_log_return_target, build_price_limits};
pub use universe::{build_cap_weighted_universe, calculate_index_weights};

#[cfg(feature = "python")]
pub use ic::{IcStats, ic_series, ic_stats};
#[cfg(feature = "python")]
pub use strategy::{
    INITIAL_CASH, Market, NavH, NavStats, NavTable, PosH, ScH, TARGET_OFFSET, TRADING_DAYS,
    nav_final, nav_stats, run, total_value, trim_scale,
};

/// Rows kept beyond a consumer's exact count look-back, absorbing the
/// amortized-compaction slack and any off-by-one at the window boundary.
pub const RETAIN_MARGIN: usize = 8;

/// A rank-1 cross-sectional array view port — the `[num_stocks]` panel currency.
pub type Av1 = ViewPort<ArrayValue<f64, 1>>;
/// A rank-1 cross-sectional array view handle (a `[num_stocks]` panel handle).
pub type AvH = Handle<Av1>;
