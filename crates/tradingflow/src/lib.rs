//! An operator library and event-loop driver for quantitative investment
//! research.
//!
//! A trading strategy is a static computation graph: feature extraction, model
//! prediction, portfolio optimization, trading simulation and performance
//! evaluation are all operator nodes. Writing a backtest amounts to wiring
//! together reusable operators; new operators are readily implemented. This
//! crate provides the [operators], the [data sources](sources), and
//! the timestamp-ordered [event loop](ingest) that drives them — layered on the
//! [`tradingflow_graph`] computation-graph engine and the [`tradingflow_data`]
//! data model.
//!
//! **`tradingflow` is the only dependency a strategy crate needs.** The engine's
//! graph-building vocabulary ([`PortHandle`](graph::PortHandle),
//! [`ViewPort`](graph::ViewPort), [`Segment`](graph::Segment), the combinators,
//! …) is re-exported as [`tradingflow::graph`](graph) and the fusion macro as
//! [`tradingflow::segment!`](segment); the data model ([`Array`] / [`Series`] /
//! [`Instant`]) as [`tradingflow::data`](data).
//!
//! # Usage
//!
//! The three things you do in every program: create a [`Scenario`], register
//! sources and operators, then [`run()`](Session::run) the event loop. Below is
//! a tiny example that records a synthetic price series, takes a rolling mean,
//! and prints its tail.
//!
//! ```rust
//! use tradingflow::{Array, Instant, Scenario, Series, SeriesView, WallClock};
//! use tradingflow::operators::{ma, record};
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
//!     let prices = sc.add_source(ArraySource::new(
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
//!     println!("{:?}", series.values());
//! }
//! ```
//!
//! The resulting computation graph:
//!
//! ```mermaid
//! flowchart LR
//!     start(( )):::hidden
//!     prices(["Array<br/><code>prices</code>"]):::array
//!     mean(["Array<br/><code>mean</code>"]):::array
//!     ma_history(["Series<br/><code>ma_history</code>"]):::series
//!     start -- ArraySource --> prices
//!     prices -- "ma(10) = Record → RollingMean, fused" --> mean
//!     mean -- Record --> ma_history
//!     classDef hidden fill:#fff,stroke:#fff,color:#fff
//!     classDef array fill:#e6f2ff,stroke:#3b82f6,color:#1e3a8a
//!     classDef series fill:#f5f5f5,stroke:#6b7280,color:#111827
//! ```
//!
//! This is the whole pattern. An actual strategy can contain many more operators
//! — [`ForwardAdjust`](operators::ForwardAdjust), `LinearRegression`,
//! `Shrinkage`, `MeanVariancePortfolio`,
//! [`RandomTrader`](operators::RandomTrader),
//! [`SharpeRatio`](operators::SharpeRatio) — but the structure stays the same.
//!
//! Every [`Array`]- or [`Series`]-shaped edge is a **view edge**: it carries a
//! borrowed view by value ([`ArrayPort`](operators::ArrayPort) /
//! [`SeriesPort`](operators::SeriesPort)), sources included — a source cell
//! lends a view of the value it owns, so there are no owned↔view bridge
//! operators to wire. Every segment has a lowercase free constructor —
//! [`ma(10)`](operators::ma), [`winsorize(p)`](operators::winsorize),
//! [`record()`](operators::record) — whose `T` / `N` generics are inferred from
//! the wiring; formula-shaped signals compose with the
//! [`tradingflow::segment!`](segment) macro, so a fused node carries no type
//! annotations beyond its parameters. Windowed operators
//! ([`ma`](operators::ma) / [`lag`](operators::lag) /
//! [`record`](operators::record)) never take a clock: **event time is ambient**,
//! carried as the engine's [graph-level context](graph::Segment::Context) and
//! advanced by the driver before each step. Behind the `python` feature,
//! operators can additionally be written in Python and run on an embedded
//! interpreter, giving strategies direct access to the Python data-science
//! ecosystem.

pub mod data;
pub mod graph;
pub mod ingest;
pub mod operators;
pub mod sources;
pub mod utils;

pub use data::{
    Array, ArrayView, Duration, Instant, Retention, Scalar, Series, SeriesView, Shape, tai_to_utc,
    utc_to_tai,
};
pub use ingest::{Scenario, Session, WallClock};
pub use utils::Schema;
