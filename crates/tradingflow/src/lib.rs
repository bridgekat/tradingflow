//! A lightweight framework for quantitative investment research.
//!
//! A trading strategy is a static computation graph: feature extraction,
//! model prediction, portfolio optimization, trading simulation and
//! performance evaluation are all operator nodes in this graph.
//! Writing a strategy backtest amounts to wiring together reusable operators,
//! and new operators can be readily implemented.
//!
//! # Basic usage
//!
//! The three things you do in every program: create a [`Scenario`], register
//! sources and operators, then use [`Scenario::build`] to create a [`Session`]
//! which runs the event loop via [`Session::run`].
//!
//! Below is a tiny example that records a synthetic price series, takes a
//! rolling mean, and prints the resulting series.
//!
//! ```rust
//! use tradingflow::operators::{rolling::*, structural::*};
//! use tradingflow::sources::{basic::*};
//! use tradingflow::{Array, ArrayPort, Instant, Scenario, Series, SeriesPort, WallClock};
//! use tradingflow_macros::segment;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Example data: a random-walk daily price series.
//!     let timestamps = (0..90).map(Instant::from_nanos).collect();
//!     let values = (0..90)
//!         .map(|_| rand::random_range(-1.0..1.0))
//!         .scan(100.0, |sum, delta| {
//!             *sum += delta;
//!             Some(*sum)
//!         })
//!         .collect();
//!     let data = Series::from_vec([], timestamps, values);
//!
//!     // Create a computation graph.
//!     let mut sc = Scenario::new(WallClock);
//!
//!     // Build the graph.
//!     let prices = sc.source(array_source(data, Array::scalar(0.0)));
//!     let prices_series = sc.segment(record(), prices);
//!     let mean = sc.segment(rolling_mean(Window::Count(10)), prices_series);
//!     let mean_series = sc.segment(record(), mean);
//!
//!     // Alternatively, one can use `segment!` to fuse operators into a
//!     // single segment (subgraph).
//!     let mean_series_fuse = sc.segment(
//!         segment!(
//!             |prices: ArrayPort<f64, 0>| -> SeriesPort<f64, 0> {
//!                 let prices_series = record() @ prices;
//!                 let mean = rolling_mean(Window::Count(10)) @ prices_series;
//!                 let mean_series = record() @ mean;
//!                 mean_series
//!             }
//!         ),
//!         prices,
//!     );
//!
//!     // Run the event loop until all sources are exhausted.
//!     let mut session = sc.build();
//!     session.run(|_, _| {}).await;
//!
//!     // Inspect results.
//!     assert_eq!(
//!         session.view(mean_series).data(),
//!         session.view(mean_series_fuse).data()
//!     );
//!     println!("{:?}", session.view(mean_series).data());
//! }
//! ```
//!
//! This is the whole pattern. An actual strategy can contain many more
//! operators — [`forward_adjust`](operators::stocks::forward_adjust),
//! [`random_trader`](operators::traders::random_trader),
//! [`sharpe_ratio`](operators::metrics::sharpe_ratio)
//! — but the overall structure stays the same.
//!
//! # Arrays and series
//!
//! See module-level docs of [`data`].
//!
//! # Building computation graphs
//!
//! See module-level docs of [`graph`].
//!
//! # The event loop driver
//!
//! See module-level docs of [`ingest`].
//!
//! # Built-in sources and operators
//!
//! See module-level docs of [`sources`] and [`operators`].

pub use tradingflow_data as data;
pub use tradingflow_graph as graph;

pub mod ingest;
pub mod operators;
pub mod ports;
pub mod sources;
pub mod utils;

pub use data::{Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape};
pub use ingest::{Scenario, Session, WallClock};
pub use ports::{ArrayPort, ArrayPorts, ArrayValue, SeriesPort, SeriesPorts, SeriesValue};
pub use utils::Schema;
