//! A lightweight framework for quantitative investment research.
//!
//! A trading strategy is a static computation graph: feature extraction,
//! model prediction, portfolio optimization, trading simulation and
//! performance evaluation are all operator nodes in this graph.
//! Writing a strategy backtest amounts to wiring together reusable operators,
//! and new operators can be readily implemented.
//!
//! # Usage
//!
//! The three things you do in every program: create a [`Scenario`], register
//! sources and operators, then use [`Scenario::build`] to create a
//! [`Session`] which runs the event loop via [`Session::run`].
//!
//! Below is a tiny example that records a synthetic price series, takes a
//! rolling mean, and prints its tail.
//!
//! ```rust
//! use tradingflow::{Array, Instant, Scenario, Series, SeriesView, WallClock};
//! use tradingflow::operators::formula::ma;
//! use tradingflow::operators::structural::record;
//! use tradingflow::sources::ArraySource;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Example data: a random-walk daily price series.
//!     let timestamps: Vec<_> = (0..90).map(Instant::from_nanos).collect();
//!     let values: Vec<f64> = /* ... */ vec![100.0; 90];
//!
//!     // Build the computation graph.
//!     let mut sc = Scenario::new(WallClock);
//!     let prices = sc.source(ArraySource::new(
//!         Series::from_vec([], timestamps, values),
//!         Array::scalar(0.0),
//!     ));
//!     let mean = sc.segment(ma(10), prices);
//!     let ma_history = sc.segment(record(), mean);
//!
//!     // Run the event loop until all sources are exhausted, then inspect results.
//!     let mut session = sc.build();
//!     session.run(|_, _| {}).await;
//!     let series = session.view(ma_history);
//!     println!("{:?}", series.data());
//! }
//! ```
//!
//! This is the whole pattern. An actual strategy can contain many more operators
//! — [`forward_adjust`](operators::stocks::forward_adjust),
//! [`random_trader`](operators::traders::random_trader),
//! [`sharpe_ratio`](operators::metrics::sharpe_ratio)
//! — but the overall structure stays the same.

pub use tradingflow_data as data;
pub use tradingflow_graph as graph;

pub mod ingest;
pub mod operators;
pub mod ports;
pub mod sources;
pub mod utils;

pub use data::{
    Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape, tai_to_utc,
    utc_to_tai,
};
pub use ingest::{Scenario, Session, WallClock};
pub use utils::Schema;
