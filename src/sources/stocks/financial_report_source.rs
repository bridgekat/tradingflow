//! Historical-only source for financial report CSVs with two date columns.
//!
//! [`FinancialReportSource`] reads a CSV file containing a report date column,
//! an optional notice date column, and numeric value columns.  The event
//! timestamp for each row is the later of the report date and the notice date
//! (with a configurable fallback offset when the notice date is missing).
//!
//! When `with_report_date` is `true`, the output array is prepended with two
//! extra elements `[year, day_of_year]` derived from the report date, for use
//! by downstream annualisation operators.

use tokio::sync::mpsc;

use crate::time::{Duration, Instant};
use crate::{Array, Source};

/// Historical-only source backed by a financial report CSV file.
///
/// # Two-timestamp logic
///
/// Financial reports are characterised by two dates:
///
/// - **Report date** — the period end (e.g. 2024-03-31 for Q1 2024).
/// - **Notice date** — the date the report was publicly released.
///
/// When `use_effective_date` is `true` (the default), the event timestamp
/// is the later of the two dates for backtesting correctness — data is not
/// available until the notice date.  When the notice date is missing, a
/// configurable fallback offset (`notice_date_fallback_ns`) is added to the
/// report date.
///
/// When `use_effective_date` is `false`, the report date is used directly
/// as the event timestamp.  This is useful for analysis that should align
/// with reporting periods rather than publication dates.
///
/// # Output shape
///
/// - `with_report_date = false` → `Array<f64>` of shape `[N]` (values only).
/// - `with_report_date = true`  → `Array<f64>` of shape `[2 + N]`:
///   `[year, day_of_year, val_1, …, val_N]` where `year` and `day_of_year`
///   are derived from the report date.
///
/// Requires a tokio runtime to be active when added to a scenario.
pub struct FinancialReportSource {
    path: String,
    report_date_column: String,
    notice_date_column: String,
    value_columns: Vec<String>,
    with_report_date: bool,
    use_effective_date: bool,
    notice_date_fallback: Duration,
    is_utc: bool,
    tz_offset: Duration,
    /// Optional inclusive start bound.
    start: Option<Instant>,
    /// Optional inclusive end bound.
    end: Option<Instant>,
}

impl FinancialReportSource {
    /// Create a new financial report source.
    ///
    /// * `use_effective_date` — if `true`, the event timestamp is
    ///   `max(report_date, notice_date)`; if `false`, the report date is
    ///   used directly.
    /// * `notice_date_fallback` — offset added to the report date when the
    ///   notice date is missing (only relevant when `use_effective_date` is
    ///   `true`).
    ///
    /// Date strings are interpreted as **UTC midnight** by default.  Use
    /// [`with_timescale`](Self::with_timescale) to change the
    /// interpretation.
    pub fn new(
        path: String,
        report_date_column: String,
        notice_date_column: String,
        value_columns: Vec<String>,
        with_report_date: bool,
        use_effective_date: bool,
        notice_date_fallback: Duration,
    ) -> Self {
        Self {
            path,
            report_date_column,
            notice_date_column,
            value_columns,
            with_report_date,
            use_effective_date,
            notice_date_fallback,
            is_utc: true,
            tz_offset: Duration::ZERO,
            start: None,
            end: None,
        }
    }

    /// Set the interpretation of the date strings in the CSV.  See
    /// [`CsvSource::with_timescale`](crate::sources::CsvSource::with_timescale)
    /// for the semantics.
    pub fn with_timescale(mut self, is_utc: bool, tz_offset: Duration) -> Self {
        self.is_utc = is_utc;
        self.tz_offset = tz_offset;
        self
    }

    /// Restrict the source to the given inclusive time range.
    ///
    /// Rows outside `[start, end]` are dropped.
    pub fn with_time_range(mut self, start: Option<Instant>, end: Option<Instant>) -> Self {
        self.start = start;
        self.end = end;
        self
    }
}

/// A parsed row from the CSV.
struct Row {
    event_ts: Instant,
    report_ts: Instant,
    report_year: f64,
    report_day_of_year: f64,
    values: Vec<f64>,
}

use super::super::csv_source::{epoch_to_instant, parse_gregorian_date};

/// Read and parse the financial report CSV, returning rows sorted by event
/// timestamp.
fn read_csv(
    path: &str,
    report_date_column: &str,
    notice_date_column: &str,
    value_columns: &[String],
    use_effective_date: bool,
    notice_date_fallback: Duration,
    is_utc: bool,
    tz_offset: Duration,
) -> Result<Vec<Row>, String> {
    let mut reader =
        csv::Reader::from_path(path).map_err(|e| format!("cannot open {path}: {e}"))?;

    // Resolve column indices from headers.
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();

    let report_idx = headers
        .iter()
        .position(|h| h == report_date_column)
        .ok_or_else(|| {
            format!("report date column {report_date_column:?} not found in CSV headers")
        })?;

    let notice_idx = headers
        .iter()
        .position(|h| h == notice_date_column);

    let value_indices: Vec<usize> = value_columns
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name.as_str())
                .ok_or_else(|| format!("value column {name:?} not found in CSV headers"))
        })
        .collect::<Result<_, _>>()?;

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;

        // Parse report date.
        let report_date = parse_gregorian_date(&record[report_idx])?;

        // Parse notice date (may be empty / missing column).
        let notice_date = if let Some(ni) = notice_idx {
            let s = record[ni].trim();
            if s.is_empty() {
                None
            } else {
                Some(parse_gregorian_date(s)?)
            }
        } else {
            None
        };

        let report_ts = epoch_to_instant(report_date, is_utc, tz_offset);
        let event_ts = if use_effective_date {
            // Effective date = max(report_date, notice_date).
            // When notice_date is missing, fall back to report_date + offset.
            let notice_ts = match notice_date {
                Some(d) => epoch_to_instant(d, is_utc, tz_offset),
                None => report_ts + notice_date_fallback,
            };
            report_ts.max(notice_ts)
        } else {
            report_ts
        };

        // Parse value columns.
        let mut values = Vec::with_capacity(value_indices.len());
        for &vi in &value_indices {
            let s = record[vi].trim();
            let v: f64 = if s.is_empty() {
                f64::NAN
            } else {
                s.parse::<f64>()
                    .map_err(|e| format!("cannot parse {s:?} as f64: {e}"))?
            };
            values.push(v);
        }

        let (year, day_of_year) = report_date.year_days_of_year();
        rows.push(Row {
            event_ts,
            report_ts,
            report_year: year as f64,
            // hifitime's day_of_year is 0-based (Jan 1 = 0); shift to 1-based
            // to match the chrono `ordinal()` convention the operators rely on.
            report_day_of_year: day_of_year + 1.0,
            values,
        });
    }

    // Sort by event timestamp (stable, preserves row order for ties).
    rows.sort_by_key(|r| r.event_ts);

    if use_effective_date {
        // Drop retrospective updates: when sorted by effective date, any
        // report whose report_date does not advance the high-water mark
        // is a late-arriving correction for an already-seen period and
        // should be ignored.
        let mut max_report_ts = Instant::MIN;
        rows.retain(|r| {
            if r.report_ts > max_report_ts {
                max_report_ts = r.report_ts;
                true
            } else {
                false
            }
        });
    }

    Ok(rows)
}

impl Source for FinancialReportSource {
    type Event = Array<f64>;
    type Output = Array<f64>;

    fn estimated_event_count(&self) -> Option<usize> {
        use super::super::csv_source::{estimate_csv_rows, scale_rows_to_range};
        let total = estimate_csv_rows(&self.path)?;
        // Scale by the report_date column (sorted in file order).  When
        // `use_effective_date` is set, add `notice_date_fallback` to the
        // user-supplied bounds so the comparison lives on the report-date
        // timeline; this mirrors the fallback applied per row when the
        // notice date is missing, and approximates the typical 1-quarter
        // lag between period end and publication.
        let shift = if self.use_effective_date {
            self.notice_date_fallback
        } else {
            Duration::ZERO
        };
        Some(scale_rows_to_range(
            total,
            &self.path,
            &self.report_date_column,
            self.start.map(|s| s - shift),
            self.end.map(|e| e - shift),
            self.is_utc,
            self.tz_offset,
            Duration::ZERO,
        ))
    }

    fn init(
        self,
        _timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, Array<f64>)>,
        mpsc::Receiver<(Instant, Array<f64>)>,
        Array<f64>,
    ) {
        let num_values = self.value_columns.len();
        let output_len = if self.with_report_date {
            2 + num_values
        } else {
            num_values
        };
        let with_report_date = self.with_report_date;
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            let rows = match read_csv(
                &self.path,
                &self.report_date_column,
                &self.notice_date_column,
                &self.value_columns,
                self.use_effective_date,
                self.notice_date_fallback,
                self.is_utc,
                self.tz_offset,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("FinancialReportSource error: {e}");
                    return;
                }
            };
            let start = self.start;
            let end = self.end;

            // When start is set, find the last row before the window
            // and emit it at start as the initial value.
            let mut last_before_start: Option<&Row> = None;
            let mut entered_window = start.is_none();

            for row in &rows {
                if let Some(s) = start {
                    if row.event_ts < s {
                        last_before_start = Some(row);
                        continue;
                    }
                }

                if !entered_window {
                    entered_window = true;
                    if let Some(init_row) = last_before_start.take() {
                        let s = start.unwrap();
                        let data = if with_report_date {
                            let mut v = Vec::with_capacity(2 + init_row.values.len());
                            v.push(init_row.report_year);
                            v.push(init_row.report_day_of_year);
                            v.extend_from_slice(&init_row.values);
                            v
                        } else {
                            init_row.values.clone()
                        };
                        let arr = Array::from_vec(&[output_len], data);
                        if hist_tx.send((s, arr)).await.is_err() {
                            break;
                        }
                    }
                }

                if let Some(e) = end {
                    if row.event_ts > e {
                        break;
                    }
                }
                let data = if with_report_date {
                    let mut v = Vec::with_capacity(2 + row.values.len());
                    v.push(row.report_year);
                    v.push(row.report_day_of_year);
                    v.extend_from_slice(&row.values);
                    v
                } else {
                    row.values.clone()
                };
                let arr = Array::from_vec(&[output_len], data);
                if hist_tx.send((row.event_ts, arr)).await.is_err() {
                    break;
                }
            }

            // If all rows were before the start, emit the last one.
            if !entered_window {
                if let Some(init_row) = last_before_start {
                    if let Some(s) = start {
                        let data = if with_report_date {
                            let mut v = Vec::with_capacity(2 + init_row.values.len());
                            v.push(init_row.report_year);
                            v.push(init_row.report_day_of_year);
                            v.extend_from_slice(&init_row.values);
                            v
                        } else {
                            init_row.values.clone()
                        };
                        let arr = Array::from_vec(&[output_len], data);
                        let _ = hist_tx.send((s, arr)).await;
                    }
                }
            }
        });

        (hist_rx, live_rx, Array::zeros(&[output_len]))
    }

    fn write(payload: Array<f64>, output: &mut Array<f64>, _timestamp: Instant) -> bool {
        output.assign(payload.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hifitime::Epoch;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// 90 days — default fallback for tests.
    const FALLBACK_90D: Duration = Duration::from_days(90);

    fn make_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn ymd(y: i32, m: u8, d: u8) -> Instant {
        Instant::from_hifitime_epoch(
            Epoch::maybe_from_gregorian_utc(y, m, d, 0, 0, 0, 0).unwrap(),
        )
    }

    fn day_of_year_1based(y: i32, m: u8, d: u8) -> f64 {
        let (_, doy) = Epoch::maybe_from_gregorian_utc(y, m, d, 0, 0, 0, 0)
            .unwrap()
            .year_days_of_year();
        doy + 1.0
    }

    #[test]
    fn parse_date_basic() {
        let d = parse_gregorian_date("2024-03-31").unwrap();
        assert_eq!(d, Epoch::maybe_from_gregorian_utc(2024, 3, 31, 0, 0, 0, 0).unwrap());
    }

    #[test]
    fn event_timestamp_uses_later_date() {
        let csv = make_csv(
            "report_date,notice_date,revenue\n\
             2024-03-31,2024-04-28,100.0\n\
             2024-06-30,2024-06-15,200.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["revenue".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        // Row 1: notice (Apr 28) > report (Mar 31) → event = Apr 28.
        let expected_0 = ymd(2024, 4, 28);
        assert_eq!(rows[0].event_ts, expected_0);

        // Row 2: report (Jun 30) > notice (Jun 15) → event = Jun 30.
        let expected_1 = ymd(2024, 6, 30);
        assert_eq!(rows[1].event_ts, expected_1);
    }

    #[test]
    fn missing_notice_date_falls_back() {
        let csv = make_csv(
            "report_date,notice_date,revenue\n\
             2024-03-31,,100.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["revenue".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        let report_ts = ymd(2024, 3, 31);
        assert_eq!(rows[0].event_ts, report_ts + FALLBACK_90D);
    }

    #[test]
    fn custom_fallback_offset() {
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,,1.0\n",
        );
        let fallback_30d: Duration = Duration::from_days(30);
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            true,
            fallback_30d,
            true,
            Duration::ZERO,
        )
        .unwrap();

        let report_ts = ymd(2024, 3, 31);
        assert_eq!(rows[0].event_ts, report_ts + fallback_30d);
    }

    #[test]
    fn use_report_date_as_event_timestamp() {
        // With use_effective_date=false, notice_date is ignored.
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,2024-08-01,100.0\n\
             2024-06-30,,200.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            false,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        // Both rows use the report date directly, sorted by report date.
        let expected_0 = ymd(2024, 3, 31);
        let expected_1 = ymd(2024, 6, 30);
        assert_eq!(rows[0].event_ts, expected_0);
        assert_eq!(rows[1].event_ts, expected_1);
    }

    #[test]
    fn retrospective_updates_dropped() {
        // Q1 report published late (Aug), Q2 report published on time (Jul).
        // Sorted by effective date: Q2 (Jul 15) then Q1 (Aug 1).
        // Q1's report_date (Mar 31) < Q2's (Jun 30) → Q1 is retrospective → dropped.
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,2024-08-01,100.0\n\
             2024-06-30,2024-07-15,200.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        // Only the Q2 report (published first) survives.
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values, vec![200.0]);
    }

    #[test]
    fn retrospective_filtering_not_applied_without_effective_date() {
        // Same data but with use_effective_date=false: both rows kept,
        // sorted by report_date.
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,2024-08-01,100.0\n\
             2024-06-30,2024-07-15,200.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            false,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values, vec![100.0]); // Q1 first by report_date
        assert_eq!(rows[1].values, vec![200.0]); // Q2 second
    }

    #[test]
    fn year_and_day_of_year() {
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,2024-04-28,1.0\n\
             2024-12-31,2025-03-01,2.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        assert_eq!(rows[0].report_year, 2024.0);
        assert_eq!(
            rows[0].report_day_of_year,
            day_of_year_1based(2024, 3, 31)
        );
        assert_eq!(rows[1].report_year, 2024.0);
        assert_eq!(
            rows[1].report_day_of_year,
            day_of_year_1based(2024, 12, 31)
        );
    }

    #[test]
    fn sorted_by_event_timestamp() {
        // CSV rows are in report_date order, but notice dates reorder them.
        // Q1 published Apr 15, Q2 published Apr 10 → sorted: Q2 first.
        let csv = make_csv(
            "report_date,notice_date,val\n\
             2024-03-31,2024-04-15,100.0\n\
             2024-06-30,2024-04-10,200.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        // Q2 notice (Apr 10) < Q1 notice (Apr 15), but Q2 event_ts =
        // max(Jun 30, Apr 10) = Jun 30 > Q1 event_ts = Apr 15.
        // Both kept (Q1 report_date < Q2 report_date, both advance).
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].values, vec![100.0]); // Q1 (event Apr 15)
        assert_eq!(rows[1].values, vec![200.0]); // Q2 (event Jun 30)
    }

    #[test]
    fn empty_value_becomes_nan() {
        let csv = make_csv(
            "report_date,notice_date,a,b\n\
             2024-03-31,2024-04-28,,5.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["a".to_string(), "b".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        assert!(rows[0].values[0].is_nan());
        assert_eq!(rows[0].values[1], 5.0);
    }

    #[test]
    fn missing_notice_column_ok() {
        let csv = make_csv(
            "report_date,val\n\
             2024-03-31,100.0\n",
        );
        let rows = read_csv(
            csv.path().to_str().unwrap(),
            "report_date",
            "notice_date",
            &["val".to_string()],
            true,
            FALLBACK_90D,
            true,
            Duration::ZERO,
        )
        .unwrap();

        let report_ts = ymd(2024, 3, 31);
        assert_eq!(rows[0].event_ts, report_ts + FALLBACK_90D);
        assert_eq!(rows[0].values, vec![100.0]);
    }
}
