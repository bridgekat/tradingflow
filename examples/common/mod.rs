//! Shared scaffolding for the A-shares cross-sectional examples.
//!
//! Pulled in by each strategy/plot example via `#[path = "common/mod.rs"] mod
//! common;`. It uses no Python operators (builds with the default feature set):
//! it builds the data pipeline (columnar parquet panel + financial-report
//! sources, the stacked cross-sectional panel, the feature sets, the
//! cap-weighted universe, the log-return target, and price limits) entirely from
//! native operators. Examples add their `tradingflow` predictor/portfolio operators
//! (and any cvxpy ones) on top.
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`args`] | CLI args (`CommonArgs`), date parsing, the symbol list |
//! | [`date`] | civil-date ↔ day-number ↔ `Instant` helpers |
//! | [`data`] | parquet sources → the stacked `[num_stocks]` panel (`Stacked`) |
//! | [`features`] | the CICC factor catalogs (fundamental + price-volume) and the feature panel over them |
//! | [`universe`] | universe masks and the cap-weighted index |
//! | [`target`] | the log-return prediction target and price limits |
//! | [`backtest`] | the decile-backtest harness (needs the `python` feature) |
//! | [`report`] | progress bar and CSV output |
//!
//! Every item is re-exported here, so examples keep using `common::*` paths.
//!
//! Differences from the Python original, all intentional:
//!
//! * Calendar/timezone handling is the caller's job in Rust (the core's
//!   timestamps are naive nanosecond counts), so rebalance dates are generated
//!   from a plain `Duration::from_days` step, and date strings are turned into
//!   `Instant`s via the [`date`] helpers — a midnight day-number parse matching
//!   `ParquetPanelSource`'s `date32` convention.
//! * The Python lambdas (`calculate_index_weights`, cross-sectional demean,
//!   price-limit rounding) become native Rust closures fed to `Map`/`Apply`.
//! * All symbols in `symbol_list.csv` are loaded (matching the original);
//!   selection of the top-`index_size` constituents happens in-graph.

#![allow(dead_code)] // not every example uses every helper
#![allow(unused_imports)] // the re-exports below serve all examples, not each one

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
pub mod date;
pub mod features;
pub mod report;
pub mod target;
pub mod universe;

pub use args::{CommonArgs, load_symbols, parse_date_days};
pub use data::{Stacked, build_stacked};
pub use date::{civil_from_days, date_str, days_from_civil, instant_from_days};
pub use features::{Features, build_features};
pub use report::{progress, read_scalar_series, write_long_csv, write_wide_csv};
pub use target::{build_log_return_target, build_price_limits};
pub use universe::{build_cap_weighted_universe, calculate_index_weights};

#[cfg(feature = "python")]
pub use ic::{IcStats, ic_series, ic_stats};
#[cfg(feature = "python")]
pub use strategy::{
    INITIAL_CASH, Market, NavStats, NavTable, TRADING_DAYS, nav_final, nav_stats, run, total_value,
    trim_scale,
};
