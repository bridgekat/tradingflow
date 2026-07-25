//! A lightweight framework for quantitative investment research.
//!
//! A trading strategy is a static computation graph: feature extraction,
//! model prediction, portfolio optimization, trading simulation and
//! performance evaluation are all operator nodes in this graph.
//! Writing a strategy backtest amounts to wiring together reusable operators,
//! and new operators can be readily implemented.
//!
//! # Examples
//!
//! The three things you do in every program: create a [`Builder`](graph::Builder),
//! register sources and operators, then use [`build`](graph::Builder::build) to
//! create a [`Graph`](graph::Graph) which runs the event loop via
//! [`run`](graph::Graph::run).
//!
//! Below is a tiny example that records a synthetic price series, takes a
//! rolling mean, and prints the resulting series.
//!
//! ```rust
//! use tradingflow::clock::UnixClock;
//! use tradingflow::data::*;
//! use tradingflow::graph::*;
//! use tradingflow::operators::{rolling, series};
//! use tradingflow::ports::*;
//! use tradingflow::segment;
//! use tradingflow::sources::basic::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Example data: a random-walk daily price series.
//!     let timestamps = (0..90)
//!         .map(|ns| Instant::from_offset(Duration::from_nanos(ns)))
//!         .collect();
//!     let values = (0..90)
//!         .map(|_| rand::random_range(-1.0..1.0))
//!         .scan(100.0, |sum, delta| {
//!             *sum += delta;
//!             Some(*sum)
//!         })
//!         .collect();
//!     let data = Series::from_parts([], timestamps, values, 0);
//!
//!     // Create the thread pool.
//!     let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//!     // Build the graph.
//!     let mut b = Builder::new(UnixClock);
//!     let prices = b.source(array_source(Array::scalar(0.0), data));
//!     let mean = b.segment(rolling::mean(10, 1), prices);
//!     let mean_series = b.segment(series::record_all(), mean);
//!
//!     // Alternatively, one can use `segment!` to fuse operators into a
//!     // single segment.
//!     let mean_series_fuse = b.segment(
//!         segment!(
//!             |prices: ArrayPort<f64, 0>| -> SeriesPort<f64, 0> {
//!                 let mean = rolling::mean(10, 1) @ prices;
//!                 let mean_series = series::record_all() @ mean;
//!                 mean_series
//!             }
//!         ),
//!         prices,
//!     );
//!
//!     // Finalize the graph.
//!     let mut g = b.build();
//!
//!     // Run the event loop until all sources are exhausted.
//!     g.run(&mut pool, |_, _| {}).await;
//!
//!     // Inspect results.
//!     assert_eq!(g.view(mean_series), g.view(mean_series_fuse));
//!     println!("{:?}", g.view(mean_series));
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
//! The [`ports`] module provides type aliases for common port types
//! (`()`, arrays, series).
//!
//! # Built-in sources and operators
//!
//! See module-level docs of [`sources`] and [`operators`].

pub use tradingflow_data as data;
pub use tradingflow_graph as graph;
pub use tradingflow_macros as macros;

pub mod clock;
pub mod operators;
pub mod ports;
pub mod sources;

pub use macros::segment;

#[cfg(feature = "python")]
pub mod python;
