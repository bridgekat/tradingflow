//! End-to-end demo: a Rust-driven graph using **Python (`flowops`) operators**,
//! writing results to CSV for plotting with matplotlib (`examples/plot.py`).
//!
//! This is the "compute in Rust, plot in Python" pattern: the whole computation
//! runs on the flow engine (Python operators on a single embedded interpreter),
//! results are recorded into `Series` and dumped to CSV, and a small standalone
//! matplotlib script renders them — so the example binary needs no plotting/GUI
//! dependency.
//!
//! Run (`pyflow` needs a venv with NumPy; a standard GIL venv is fine):
//!
//! ```text
//! set PYO3_PYTHON=...\.venv\Scripts\python.exe
//! set PATH=...\pythoncore-3.13-64;%PATH%               # python3.13.dll
//! set PYTHONPATH=...\python;...\.venv\Lib\site-packages   # flowops + numpy
//! cargo run --example flowops_demo --features pyflow
//! python examples/plot.py target/flowops_demo.csv      # render with matplotlib
//! ```

use std::fmt::Write as _;
use std::fs;

use flowgraph::core::Pool;
use flowgraph::typed::{Graph, GraphBuilder, RefPort, RefPorts, RefSource};

use tradingflow::operators::{Clock, PyClassOperator, PyParams, Record};
use tradingflow::{Array, Instant, Series};

fn main() {
    const N: usize = 5; // stocks
    const F: usize = 3; // features
    const T: usize = 60; // ticks

    let clock = Clock::new();
    let mut b = GraphBuilder::new();

    // Sources (driven manually below): universe + raw feature/return feeds.
    let universe = b.push_source(RefSource::new(Array::from_vec(&[N], vec![1.0; N])));
    let feat_feed = b.push_source(RefSource::new(Array::zeros(&[N, F])));
    let tgt_feed = b.push_source(RefSource::new(Array::zeros(&[N])));

    // Record the feeds into Series so the predictor can read history.
    let feat_series = b.push(Record::new(clock.clone()), *feat_feed);
    let tgt_series = b.push(Record::new(clock.clone()), *tgt_feed);

    // Python operator: a ridge cross-sectional mean predictor (flowops).
    let pred = b.push(
        PyClassOperator::<(RefPort<Array<f64>>, RefPort<Series<f64>>, RefPort<Series<f64>>)>::from_module(
            "flowops.predictors.mean.incremental_ridge",
            PyParams::new()
                .int("num_stocks", N as i64)
                .int("num_features", F as i64)
                .int("universe_size", N as i64)
                .int("target_offset", 1)
                .float("alpha", 1.0),
            vec![N],
            clock.clone(),
        ),
        (*universe, feat_series, tgt_series),
    );

    // Python operator: turnover of the prediction vector (flowops metric).
    let turnover = b.push(
        PyClassOperator::<RefPorts<Array<f64>>>::from_module(
            "flowops.metrics.turnover",
            PyParams::new().int("num_stocks", N as i64),
            vec![],
            clock.clone(),
        ),
        &[pred][..],
    );

    // Record the metric so we can dump a time series.
    let turnover_series = b.push(Record::new(clock.clone()), turnover);

    let mut g = Graph::from_builder(b);
    let mut pool = Pool::new(0);

    // Drive synthetic ticks: features carry a noisy linear signal of the target.
    for t in 1..=T as i64 {
        let phase = t as f64 * 0.3;
        let x: Vec<f64> = (0..N * F)
            .map(|k| (phase + 0.7 * (k as f64)).sin())
            .collect();
        let y: Vec<f64> = (0..N)
            .map(|i| (phase + i as f64).sin() * 0.5)
            .collect();
        clock.set(Instant::from_nanos(t * 86_400_000_000_000)); // ~daily
        *g.state_mut(feat_feed) = Array::from_vec(&[N, F], x);
        *g.state_mut(tgt_feed) = Array::from_vec(&[N], y);
        *g.state_mut(universe) = Array::from_vec(&[N], vec![1.0; N]);
        g.stabilize(&mut pool);
    }

    // Read the recorded turnover series and write a tidy CSV.
    let series: &Series<f64> = g.ref_view(turnover_series);
    let mut csv = String::from("timestamp_ns,turnover\n");
    for (ts, v) in series.timestamps().iter().zip(series.values().iter()) {
        writeln!(csv, "{},{}", ts.as_nanos(), v).unwrap();
    }
    let path = "target/flowops_demo.csv";
    fs::write(path, csv).expect("write csv");

    println!(
        "wrote {} rows to {path}\nplot with:  python examples/plot.py {path}",
        series.len()
    );
}
