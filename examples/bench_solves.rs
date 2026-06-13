//! Microbenchmark: does the flow `Pool` parallelize K independent cvxpy
//! Markowitz solves? Isolates solve parallelism from any data/predictor
//! backbone — one tick fans out to K Markowitz portfolio nodes that all read
//! the same fixed (mu, Sigma), repeated over T ticks.
//!
//! Run (GIL venv with cvxpy):
//! ```text
//! bench_solves.exe --n 300 --k 8 --ticks 40 --threads 0   # serial
//! bench_solves.exe --n 300 --k 8 --ticks 40 --threads 8   # parallel
//! ```

use std::time::Instant as Wall;

use flowgraph::typed::RefPort;

use tradingflow::operators::{PyClassOperator, PyParams, Resample};
use tradingflow::Scenario;
use tradingflow::sources::clock;
use tradingflow::{Array, Instant};

const MODE_MIN_MEAN_VARIANCE: i64 = 3;

fn arg(name: &str, def: usize) -> usize {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn arg_str(name: &str, def: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned().unwrap_or_else(|| def.to_string())
}

#[tokio::main]
async fn main() {
    let n = arg("--n", 300);
    let k = arg("--k", 8);
    let ticks = arg("--ticks", 40);
    let threads = arg("--threads", 0);
    let module = arg_str("--module", "flowops.portfolios.mean_variance.markowitz");

    // Fixed PSD covariance Sigma = A Aᵀ / n + I and mean mu (deterministic).
    let a: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.12345).sin()).collect();
    let mut sigma = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for t in 0..n {
                s += a[i * n + t] * a[j * n + t];
            }
            sigma[i * n + j] = s / n as f64 + if i == j { 1.0 } else { 0.0 };
        }
    }
    let mu: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.777).cos() * 0.01).collect();

    let mut sc = Scenario::new();
    let clk = sc.clock();

    // T-tick clock; universe re-emitted each tick so every Markowitz fires.
    let ticks_v: Vec<Instant> = (1..=ticks as i64).map(|t| Instant::from_nanos(t * 86_400_000_000_000)).collect();
    let clock_h = sc.add_source(clock(ticks_v), ());
    let ones = sc.add_const(Array::from_vec(&[n], vec![1.0; n]));
    let universe = sc.add_operator(Resample::<Array<f64>, ()>::new(), (clock_h, *ones));

    let mu_h = sc.add_const(Array::from_vec(&[n], mu));
    let sigma_h = sc.add_const(Array::from_vec(&[n, n], sigma));

    // K independent Markowitz portfolios (distinct delta), all on the same tick.
    let mut recs = Vec::new();
    for i in 0..k {
        let delta = 1.0 + i as f64;
        let mk = sc.add_operator(
            PyClassOperator::<(RefPort<Array<f64>>, RefPort<Array<f64>>, RefPort<Array<f64>>)>::from_module(
                &module,
                PyParams::new()
                    .int("num_stocks", n as i64)
                    .int("mode", MODE_MIN_MEAN_VARIANCE)
                    .float("bound", delta)
                    .bool("long_only", true),
                vec![n],
                clk.clone(),
            ),
            (universe, *mu_h, *sigma_h),
        );
        recs.push(sc.add_record(mk));
    }

    let t0 = Wall::now();
    let mut session = sc.build_with_threads(threads);
    session.run(|_, _| {}).await;
    let dt = t0.elapsed().as_secs_f64();

    // Touch outputs so nothing is optimized away.
    let mut checksum = 0.0;
    for r in &recs {
        let s: &tradingflow::Series<f64> = session.value(*r);
        checksum += s.values().iter().filter(|x| x.is_finite()).sum::<f64>();
    }
    let total_solves = k * ticks;
    let mod_short = module.rsplit('.').next().unwrap_or(&module);
    println!(
        "[{mod_short}] n={n} k={k} ticks={ticks} threads={threads}  solves={total_solves}  wall={dt:.2}s  ms/solve={:.1}  checksum={checksum:.3}",
        dt * 1000.0 / total_solves as f64
    );
}
