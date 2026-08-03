//! A lightweight framework for quantitative investment research.
//!
//! A trading strategy is a *static, stateful computation graph*: feature
//! extraction, model prediction, portfolio optimization, trading simulation
//! and performance evaluation are all operator nodes in this graph. When new
//! data arrives, changes are automatically propagated throughout the graph.
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
//!     graph::{Builder, Operator, Pool},
//!     operators::{array, elem, rolling, series, signal, trader},
//!     ports::{ArrayPort, SignalPort},
//!     sources::sync,
//!     time::UnixTime,
//! };
//!
//! #[tokio::main]
//! async fn main() {
//!     const N_SYMBOLS: usize = 5;
//!     const N_DAYS: usize = 30;
//!
//!     // Example data: 5 instruments, each having random-walk daily price series.
//!     let mut timestamps = Vec::new();
//!     let mut values = Vec::new();
//!     let mut prices = [100.0; N_SYMBOLS];
//!     for i in 0..N_DAYS {
//!         timestamps.push(Instant::from_offset(Duration::from_days(i as i64)));
//!         values.push(Array::from(prices));
//!         for price in &mut prices {
//!             *price += rand::random_range(-3.0..3.0);
//!         }
//!     }
//!     let data = Series::from((timestamps, values));
//!
//!     // Create the thread pool.
//!     let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());
//!
//!     // Build the graph.
//!     let mut b = Builder::new(UnixTime);
//!     let (daily, prices) = b.source(sync::array_series(data));
//!
//!     // The MACD indicator is simple enough to be implemented by composing built-in operators.
//!     let ma_fast = b.op(rolling::mean(12, 1), (daily, prices)); // MA(12) of prices
//!     let ma_slow = b.op(rolling::mean(26, 1), (daily, prices)); // MA(26) of prices
//!     let macd = b.op(elem::sub(), (ma_fast, ma_slow)); // MA(12) - MA(26)
//!     let smooth = b.op(rolling::mean(9, 1), (daily, macd)); // MA(9) of MACD
//!     let diff = b.op(elem::sub(), (macd, smooth)); // (MACD - smooth)
//!     let prev = b.op(rolling::lag(1), (daily, diff)); // (MACD - smooth) one period ago
//!
//!     // Calculate position weight based on `diff` and `prev`, on each rebalance signal.
//!     // This requires defining a custom operator, by implementing the `Operator` trait.
//!     struct Crossover;
//!
//!     impl Operator for Crossover {
//!         type Inputs = (SignalPort<0>, ArrayPort<f64, 1>, ArrayPort<f64, 1>);
//!         type Outputs = ArrayPort<f64, 1>;
//!         type Context = Instant;
//!         type State = Array<f64, 1>;
//!
//!         fn init(
//!             self,
//!             _: (
//!                 ArrayView<'_, bool, 0>,
//!                 ArrayView<'_, f64, 1>,
//!                 ArrayView<'_, f64, 1>,
//!             ),
//!         ) -> Self::State {
//!             // Initialize state (weights) to 0.0.
//!             [0.0; N_SYMBOLS].into()
//!         }
//!
//!         fn reset<'a, 'b: 'a>(
//!             _: (
//!                 ArrayView<'a, bool, 0>,
//!                 ArrayView<'a, f64, 1>,
//!                 ArrayView<'a, f64, 1>,
//!             ),
//!             state: &'b mut Self::State,
//!         ) -> ArrayView<'a, f64, 1> {
//!             // Output current state (weights).
//!             state.view()
//!         }
//!
//!         fn compute<'a, 'b: 'a>(
//!             (rebalance_signal, diff, prev): (
//!                 ArrayView<'a, bool, 0>,
//!                 ArrayView<'a, f64, 1>,
//!                 ArrayView<'a, f64, 1>,
//!             ),
//!             state: &'b mut Self::State,
//!             _: &Self::Context,
//!         ) -> ArrayView<'a, f64, 1> {
//!             // Update state (weights) only when rebalance signal is true.
//!             if *rebalance_signal {
//!                 for (i, (&diff, &prev)) in diff.iter().zip(prev.iter()).enumerate() {
//!                     // Buy: set weight for the i-th symbol to (1 / N).
//!                     if diff > 0.0 && prev <= 0.0 {
//!                         state[[i]] = 1.0 / N_SYMBOLS as f64;
//!                     }
//!                     // Sell: set weight for the i-th symbol to 0.
//!                     if diff < 0.0 && prev >= 0.0 {
//!                         state[[i]] = 0.0;
//!                     }
//!                 }
//!             }
//!             // Output current state (weights).
//!             state.view()
//!         }
//!     }
//!
//!     // Wire the custom operator into the graph.
//!     // Rebalance frequency is set to daily.
//!     let weights = b.op(Crossover, (daily, diff, prev));
//!
//!     // Simulate frictionless trading using `weight`.
//!     // Here we assume: best bid = best ask = prices, no dividends.
//!     let flags = b.val(array::constant([true; N_SYMBOLS]));
//!     let bids = prices;
//!     let asks = prices;
//!     let div_signals = b.val(signal::quiet([N_SYMBOLS]));
//!     let share_divs = b.val(array::constant([0.0; N_SYMBOLS]));
//!     let cash_divs = b.val(array::constant([0.0; N_SYMBOLS]));
//!
//!     let (_positions, _cash, nav) = b.op(
//!         trader::fixed::benchmark(false, 100.0),
//!         (
//!             (daily, flags, bids, asks),
//!             (div_signals, share_divs, cash_divs),
//!             (daily, weights),
//!         ),
//!     );
//!     let nav_series = b.op(series::record_all(), (daily, nav));
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
//! [`trader::fixed::benchmark`](operators::trader::fixed::benchmark),
//! [`metric::performance::return_sharpe`](operators::metric::performance::return_sharpe)
//! — but the overall structure stays the same.
//!
//! # Arrays and series
//!
//! See module-level docs of [`data`].
//!
//! # Building computation graphs
//!
//! See module-level docs of [`graph`] and [`ports`].
//!
//! The [`ports`] module provides additional type aliases for [`graph::Port`]
//! which are more commonly used for trading strategies: instead of passing
//! around scalars or thin references, most operators here pass boolean signal
//! arrays, scalar value arrays or time series by their views.
//!
//! # Built-in data sources
//!
//! See [`sources`] and its sub-modules.
//!
//! # Built-in operator nodes
//!
//! See [`operators`] and its sub-modules.
//!
//! # Factor expressions
//!
//! See module-level docs of [`expr`].
//!
//! # Python operators
//!
//! See module-level docs of [`python`].

pub use tradingflow_data as data;
pub use tradingflow_graph as graph;
pub use tradingflow_macros as macros;

pub mod expr;
pub mod operators;
pub mod ports;
pub mod sources;
pub mod time;

pub use macros::fuse;

#[cfg(feature = "python")]
pub mod python;
