//! MACD crossover strategy over synthetic data.

use tradingflow::{
    data::{Array, Duration, Instant, Series},
    graph::{Builder, Pool},
    operators::{array, elem, rolling, series, signal, trader},
    sources::sync,
    time::UnixTime,
};

#[tokio::main]
async fn main() {
    const N_SYMBOLS: usize = 5;
    const N_DAYS: usize = 30;

    // Example data: 5 instruments, each having random-walk daily price series.
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    let mut prices = [100.0; N_SYMBOLS];
    for i in 0..N_DAYS {
        timestamps.push(Instant::from_offset(Duration::from_days(i as i64)));
        values.push(Array::from(prices));
        for price in &mut prices {
            *price += rand::random_range(-3.0..3.0);
        }
    }
    let data = Series::from((timestamps, values));

    // Create the thread pool.
    let mut pool = Pool::new(std::thread::available_parallelism().unwrap().get());

    // Build the graph.
    let mut b = Builder::new(UnixTime);
    let (daily, prices) = b.source(sync::array_series(data));

    // The MACD indicator is simple enough to be implemented by composing built-in operators.
    let ema_fast = b.op(rolling::mean_exp(12, 1), (daily, prices)); // EMA(12) of prices
    let ema_slow = b.op(rolling::mean_exp(26, 1), (daily, prices)); // EMA(26) of prices
    let macd = b.op(elem::sub(), (ema_fast, ema_slow)); // EMA(12) - EMA(26)
    let smooth = b.op(rolling::mean_exp(9, 1), (daily, macd)); // EMA(9) of MACD
    let held = b.op(elem::gt(), (macd, smooth)); // MACD > smooth
    let weights = b.op(elem::indicator(1.0 / N_SYMBOLS as f64, 0.0), held); // 1/N position if held

    // Simulate frictionless trading using `weight`.
    // Here we assume: best bid = best ask = prices, no dividends.
    // Execution is delayed by one period after the rebalance signal.
    let flags = b.val(array::constant([true; N_SYMBOLS]));
    let (bids, asks) = (prices, prices);
    let div_signals = b.val(signal::quiet([N_SYMBOLS]));
    let share_divs = b.val(array::constant([0.0; N_SYMBOLS]));
    let cash_divs = b.val(array::constant([0.0; N_SYMBOLS]));

    let (_positions, _cash, nav) = b.op(
        trader::fixed::benchmark(true, 100.0),
        (
            (daily, flags, bids, asks),
            (div_signals, share_divs, cash_divs),
            (daily, weights),
        ),
    );
    let nav_series = b.op(series::record_all(), (daily, nav));

    // Finish building the graph.
    let mut g = b.build();

    // Run the event loop until all sources are exhausted.
    g.run(&mut pool, |_, _| {}).await;

    // Inspect results.
    println!("{:?}", g.view(nav_series).to_contiguous());
}
