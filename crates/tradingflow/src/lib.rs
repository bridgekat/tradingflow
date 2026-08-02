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
//! Below is a tiny example that generates a synthetic price series, simulates
//! trading using the MACD crossover strategy, and prints the resulting net
//! asset value series.
//!
//! ```rust
//! use tradingflow::{
//!     data::{Array, ArrayView, Duration, Instant, Series},
//!     graph::{Builder, Pool, Segment},
//!     operators::{array, elem, rolling, series, signal, trader},
//!     ports::ArrayPort,
//!     sources::sync,
//!     time::UnixTime,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     // Example data: a random-walk daily price series.
//!     let mut timestamps = Vec::new();
//!     let mut values = Vec::new();
//!     let mut price = 100.0;
//!     for i in 0..30 {
//!         timestamps.push(Instant::from_offset(Duration::from_days(i)));
//!         values.push(price);
//!         price += rand::random_range(-1.0..1.0);
//!     }
//!     let data = Series::from_parts([], timestamps, values, 0);
//!
//!     // Create the thread pool.
//!     let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//!     // Build the graph.
//!     let mut b = Builder::new(UnixTime);
//!     let (daily, prices) = b.source(sync::array_series(data));
//!
//!     // The MACD indicator is simple enough to be implemented by composing built-in operators.
//!     let ma_fast = b.segment(rolling::mean(12, 1), (daily, prices)); // MA(12) of prices
//!     let ma_slow = b.segment(rolling::mean(26, 1), (daily, prices)); // MA(26) of prices
//!     let macd = b.segment(elem::sub(), (ma_fast, ma_slow)); // MA(12) - MA(26)
//!     let smooth = b.segment(rolling::mean(9, 1), (daily, macd)); // MA(9) of MACD
//!     let diff = b.segment(elem::sub(), (macd, smooth)); // (MACD - smooth)
//!     let prev = b.segment(rolling::lag(1), (daily, diff)); // (MACD - smooth) one period ago
//!
//!     // Calculate position weight based on `diff` and `prev`.
//!     // This requires defining a custom operator, by implementing the `Segment` trait.
//!     struct Crossover;
//!
//!     impl Segment for Crossover {
//!         type Inputs = (ArrayPort<f64, 0>, ArrayPort<f64, 0>);
//!         type Outputs = ArrayPort<f64, 0>;
//!         type Context = Instant;
//!         type State = Array<f64, 0>;
//!
//!         fn init(self, _: (ArrayView<'_, f64, 0>, ArrayView<'_, f64, 0>)) -> Self::State {
//!             Array::scalar(0.0) // Initialize state (weight) to 0.0.
//!         }
//!
//!         fn reset<'a, 'b: 'a>(
//!             _: (ArrayView<'a, f64, 0>, ArrayView<'a, f64, 0>),
//!             state: &'b mut Self::State,
//!         ) -> ArrayView<'a, f64, 0> {
//!             state.view() // Output current state (weight).
//!         }
//!
//!         fn compute<'a, 'b: 'a>(
//!             (diff, prev): (ArrayView<'a, f64, 0>, ArrayView<'a, f64, 0>),
//!             state: &'b mut Self::State,
//!             _: &Self::Context,
//!         ) -> ArrayView<'a, f64, 0> {
//!             if *diff > 0.0 && *prev <= 0.0 {
//!                 **state = 1.0; // Buy signal: set state (weight) to 1.0.
//!             }
//!             if *diff < 0.0 && *prev >= 0.0 {
//!                 **state = 0.0; // Sell signal: set state (weight) to 0.0.
//!             }
//!             state.view() // Output current state (weight).
//!         }
//!     }
//!
//!     // Wire the custom operator into the graph.
//!     let weight = b.segment(Crossover, (diff, prev));
//!
//!     // Simulate frictionless trading using `weight`.
//!     // Here we assume: best bid = best ask = prices, no dividends.
//!     let price_signal = daily;
//!     let flags = b.value(array::from_parts([1], vec![true].into()));
//!     let bids = b.segment(array::pad_ndim(), prices); // Convert scalar to 1D vector
//!     let asks = b.segment(array::pad_ndim(), prices); // (Same)
//!
//!     let div_signals = b.segment(signal::as_signal_map(array::pad_ndim()), daily);
//!     let share_divs = b.value(array::from_parts([1], vec![0.0].into()));
//!     let cash_divs = b.value(array::from_parts([1], vec![0.0].into()));
//!
//!     let rebalance_signal = daily;
//!     let weights = b.segment(array::pad_ndim(), weight);
//!
//!     let (_positions, _cash, nav) = b.segment(
//!         trader::fixed::benchmark(false, 100.0),
//!         (
//!             (price_signal, flags, bids, asks),
//!             (div_signals, share_divs, cash_divs),
//!             (rebalance_signal, weights),
//!         ),
//!     );
//!     let nav_series = b.segment(series::record_all(), (daily, nav));
//!
//!     // Finish building the graph.
//!     let mut g = b.build();
//!
//!     // Run the event loop until all sources are exhausted.
//!     g.run(&mut pool, |_, _| {}).await;
//!
//!     // Inspect results.
//!     println!("{:?}", g.view(nav_series).to_contiguous());
//! }
//! ```
//!
//! This is the whole pattern. An actual strategy can contain many more
//! operators — [`feature::forward_adjust`](operators::feature::forward_adjust),
//! [`trader::fixed::random`](operators::trader::fixed::random),
//! [`metric::performance::return_sharpe`](operators::metric::performance::return_sharpe)
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
//! (booleans, arrays, series).
//!
//! # Built-in sources and operators
//!
//! See module-level docs of [`sources`] and [`operators`].

pub use tradingflow_data as data;
pub use tradingflow_graph as graph;
pub use tradingflow_macros as macros;

pub mod operators;
pub mod ports;
pub mod sources;
pub mod time;

pub use macros::segment;

#[cfg(feature = "python")]
pub mod python;
