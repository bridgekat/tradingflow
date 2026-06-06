//! Pure-native microbenchmark for the data-load + event-loop path.
//!
//! Builds the full stacked cross-sectional panel + the 7-factor feature set
//! from `common` (NO `pyflow`, so no cvxpy/venv needed) and runs the Session
//! event loop to completion, timing the build phase (source init + tokio
//! spawns) separately from the run phase (channel drain + heap merge + native
//! stabilize). This isolates whether the serial merge of ~29k channels + the
//! per-batch stabilize is the wall-clock bottleneck independent of any Python.
//!
//! ```text
//! cargo run --release --example bench_dataload -- --index-size 300 --threads 0
//! cargo run --release --example bench_dataload -- --index-size 300 --threads 8
//! ```
//!
//! Extra flags beyond the shared `common::Args`:
//!   --max-symbols N   cap the number of symbols loaded (default: all)
//!   --rt-threads N    tokio worker threads (default 0 = multi-thread default)

#[path = "common/mod.rs"]
mod common;

use std::time::Instant as StdInstant;

use tradingflow::flow::Scenario;

fn arg_usize(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == name {
            return args[i + 1].parse().unwrap_or(default);
        }
    }
    default
}

fn main() {
    let rt_threads = arg_usize("--rt-threads", 0);
    let mut builder = if rt_threads == 0 {
        tokio::runtime::Builder::new_multi_thread()
    } else {
        let mut b = tokio::runtime::Builder::new_multi_thread();
        b.worker_threads(rt_threads);
        b
    };
    let rt = builder.enable_all().build().unwrap();
    rt.block_on(run());
}

async fn run() {
    let args = common::Args::from_env();
    let mut symbols = common::load_symbols(&args.data_dir);
    let max_symbols = arg_usize("--max-symbols", symbols.len());
    symbols.truncate(max_symbols);
    let n = symbols.len();
    eprintln!(
        "symbols={n} index_size={} threads={} window={} days=[{}..{}]",
        args.index_size, args.threads, args.window, args.data_start_days, args.end_days
    );

    // ---- BUILD PHASE: source registration + graph build + source init/spawn.
    let t_build0 = StdInstant::now();
    let mut sc = Scenario::new();
    let st = common::build_stacked(&mut sc, &symbols, &args);
    let _features = common::build_features(&mut sc, &st, &args);
    let t_graph = t_build0.elapsed();

    let t_init0 = StdInstant::now();
    let mut session = sc.build_with_threads(args.threads);
    let t_init = t_init0.elapsed();

    // ---- RUN PHASE: drain channels, merge heap, stabilize per batch.
    let mut batches: usize = 0;
    let mut last_events: usize = 0;
    let t_run0 = StdInstant::now();
    session
        .run(|_ts, events| {
            batches += 1;
            last_events = events;
        })
        .await;
    let t_run = t_run0.elapsed();

    eprintln!(
        "graph_build={:?}  source_init+spawn={:?}  run={:?}  batches={}  events={}",
        t_graph, t_init, t_run, batches, last_events
    );
    let secs = t_run.as_secs_f64();
    if secs > 0.0 {
        eprintln!(
            "throughput: {:.0} batches/s, {:.0} events/s",
            batches as f64 / secs,
            last_events as f64 / secs
        );
    }
}
