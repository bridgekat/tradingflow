//! Terminal progress reporting and CSV output for the plot scripts.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;

use tradingflow::graph::{PortHandle, RefPort};
use tradingflow::operators::SeriesPort;

use tradingflow::data::civil_from_days;
use tradingflow::{Instant, SeriesView, Session};

/// `YYYY-MM-DD` for an event [`Instant`].
pub fn date_str(ts: Instant) -> String {
    let (y, m, d) = civil_from_days(ts.to_utc_days());
    format!("{y:04}-{m:02}-{d:02}")
}

/// A `tqdm`-style progress callback (backed by [`indicatif`]) for
/// `Session::run`'s `on_stable`.
///
/// Progress is measured in **long-table rows**: the panel sources emit one event
/// per narrow row, so the driver's `events()` count *is* the row count (no shared
/// counter needed). `total` is the session's `total_num_events()` in
/// the same row unit; `Some(n)` → a bounded bar (percent / rate / ETA), else a
/// spinner. `{per_sec}` is therefore rows/s. `begin` sets `{prefix}` (warm-up
/// before it, running after); `{msg}` is the current event date. The bar uses a
/// terminal-width `{wide_bar}` with Unicode sub-cell fill, and finalises itself
/// when the callback drops at the end of `run`:
/// ```ignore
/// let total = session.total_num_events();
/// session.run(common::progress(total, args.begin())).await;
/// eprintln!(); // move past the finished bar line before printing results
/// ```
pub fn progress(total: Option<usize>, begin: Instant) -> impl FnMut(&Session, Instant) {
    use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

    // Finish (leave) the bar when the callback is dropped — i.e. when `run`
    // returns — so the final state persists without the caller managing it.
    struct FinishOnDrop(ProgressBar);
    impl Drop for FinishOnDrop {
        fn drop(&mut self) {
            self.0.finish();
        }
    }

    let pb = match total {
        Some(t) if t > 0 => {
            let pb = ProgressBar::new(t as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "{prefix} |{wide_bar}| {pos}/{len} [{elapsed}<{eta}, {per_sec}]",
                )
                .unwrap()
                .with_key(
                    "per_sec",
                    |s: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                        write!(w, "{:.0} events/s", s.per_sec()).unwrap()
                    },
                )
                .progress_chars("█▉▊▋▌▍▎▏ "),
            );
            pb
        }
        _ => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{prefix} {spinner} {pos} [{elapsed}, {per_sec}]")
                    .unwrap()
                    .with_key(
                        "per_sec",
                        |s: &indicatif::ProgressState, w: &mut dyn std::fmt::Write| {
                            write!(w, "{:.0} events/s", s.per_sec()).unwrap()
                        },
                    ),
            );
            pb
        }
    };
    // Cap redraws at ~20 fps regardless of how often the callback fires.
    pb.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));

    let _begin_ns = begin.as_nanos();
    let guard = FinishOnDrop(pb);
    move |session: &Session, ts: Instant| {
        let pb = &guard.0;
        let rows = session.num_events() as u64;
        // Grow the length if the estimate undershot (keeps the percentage sane).
        if let Some(len) = pb.length()
            && rows > len
        {
            pb.set_length(rows);
        }
        pb.set_position(rows);
        pb.set_prefix(date_str(ts));
    }
}

/// Read a recorded **scalar** series into `(timestamps_ns, values)`.
pub fn read_scalar_series(
    session: &Session,
    h: PortHandle<SeriesPort<f64, 0>>,
) -> (Vec<i64>, Vec<f64>) {
    let s: SeriesView<f64, 0> = session.view(h);
    let ts = s.timestamps().iter().map(|t| t.as_nanos()).collect();
    let vals = s.values().to_vec();
    (ts, vals)
}

/// Write labelled scalar series in long format (`series,timestamp_ns,value`)
/// so the plot scripts can group by series and handle independent cadences.
pub fn write_long_csv(path: &str, series: &[(String, Vec<i64>, Vec<f64>)]) {
    let mut csv = String::from("series,timestamp_ns,value\n");
    for (label, ts, vals) in series {
        for (t, v) in ts.iter().zip(vals.iter()) {
            writeln!(csv, "{label},{t},{v}").unwrap();
        }
    }
    fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
}

/// Align labelled scalar series by timestamp into a wide CSV (NaN-filled).
pub fn write_wide_csv(path: &str, series: &[(String, Vec<i64>, Vec<f64>)]) {
    let ncols = series.len();
    let mut rows: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (c, (_, ts, vals)) in series.iter().enumerate() {
        for (t, v) in ts.iter().zip(vals.iter()) {
            rows.entry(*t).or_insert_with(|| vec![f64::NAN; ncols])[c] = *v;
        }
    }
    let mut csv = String::from("timestamp_ns");
    for (label, _, _) in series {
        write!(csv, ",{label}").unwrap();
    }
    csv.push('\n');
    for (t, vals) in &rows {
        write!(csv, "{t}").unwrap();
        for v in vals {
            write!(csv, ",{v}").unwrap();
        }
        csv.push('\n');
    }
    fs::write(path, csv).unwrap_or_else(|e| panic!("write {path}: {e}"));
}
