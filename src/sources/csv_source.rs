//! Historical-only source that reads a CSV file asynchronously.

use chrono::NaiveDate;
use csv_async::StringRecord;
use tokio::sync::mpsc;

use crate::{Array, Source};

/// Historical-only source backed by a CSV file.
///
/// Reads a timestamp column and a set of named value columns from a CSV
/// file on disk.  Rows are sorted by timestamp; each row becomes one
/// event carrying an `Array<f64>` of the selected columns.
///
/// Requires a tokio runtime to be active when added to a scenario.
pub struct CsvSource {
    path: String,
    time_column: String,
    value_columns: Vec<String>,
    timestamp_offset_ns: i64,
}

impl CsvSource {
    /// Create a new CSV source.
    ///
    /// * `path` — filesystem path to the CSV file.
    /// * `time_column` — header name of the date/datetime column
    ///   (parsed as `YYYY-MM-DD`).
    /// * `value_columns` — header names of columns to include as values,
    ///   in order.  Each is parsed as `f64`.
    /// * `timestamp_offset_ns` — constant offset in nanoseconds added to
    ///   every parsed timestamp before it is used as the event timestamp.
    ///   Useful when the CSV contains low-precision timestamps (e.g. dates)
    ///   that would otherwise cause forward-looking bias against
    ///   higher-precision sources.
    pub fn new(
        path: String,
        time_column: String,
        value_columns: Vec<String>,
        timestamp_offset_ns: i64,
    ) -> Self {
        Self {
            path,
            time_column,
            value_columns,
            timestamp_offset_ns,
        }
    }
}

/// Parse a date string (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) to
/// nanoseconds since the UNIX epoch (midnight UTC).
fn parse_timestamp(s: &str) -> Result<i64, String> {
    let date = if let Ok(d) = NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d") {
        d
    } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S") {
        dt.date()
    } else {
        return Err(format!("cannot parse date: {s:?}"));
    };
    date.and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .ok_or_else(|| format!("timestamp overflow for {s:?}"))
}

/// Resolve column indices from CSV headers.
fn resolve_columns(
    headers: &StringRecord,
    time_column: &str,
    value_columns: &[String],
) -> Result<(usize, Vec<usize>), String> {
    let time_idx = headers
        .iter()
        .position(|h| h == time_column)
        .ok_or_else(|| format!("time column {time_column:?} not found in CSV headers"))?;
    let value_indices: Vec<usize> = value_columns
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|h| h == name.as_str())
                .ok_or_else(|| format!("value column {name:?} not found in CSV headers"))
        })
        .collect::<Result<_, _>>()?;
    Ok((time_idx, value_indices))
}

/// Parse value columns from a CSV record.
fn parse_values(record: &StringRecord, value_indices: &[usize]) -> Result<Vec<f64>, String> {
    let mut values = Vec::with_capacity(value_indices.len());
    for &vi in value_indices {
        let s = record[vi].trim();
        let v: f64 = if s.is_empty() {
            f64::NAN
        } else {
            s.parse::<f64>()
                .map_err(|e| format!("cannot parse {s:?} as f64: {e}"))?
        };
        values.push(v);
    }
    Ok(values)
}

impl Source for CsvSource {
    type Event = Array<f64>;
    type Output = Array<f64>;

    fn init(
        self,
        _timestamp: i64,
    ) -> (
        mpsc::Receiver<(i64, Array<f64>)>,
        mpsc::Receiver<(i64, Array<f64>)>,
        Array<f64>,
    ) {
        let num_columns = self.value_columns.len();
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        // Stream rows asynchronously from the CSV file.  Each row is sent
        // through the bounded channel with back-pressure, avoiding
        // buffering the entire file in memory.
        //
        // Assumes timestamps are non-decreasing (true for crawler output
        // and most historical data).
        tokio::spawn(async move {
            let file = match tokio::fs::File::open(&self.path).await {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("CsvSource error: cannot open {}: {e}", self.path);
                    return;
                }
            };
            let mut reader = csv_async::AsyncReader::from_reader(file);
            let headers = match reader.headers().await {
                Ok(h) => h.clone(),
                Err(e) => {
                    eprintln!("CsvSource error: {e}");
                    return;
                }
            };
            let (time_idx, value_indices) =
                match resolve_columns(&headers, &self.time_column, &self.value_columns) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("CsvSource error: {e}");
                        return;
                    }
                };

            let mut record = StringRecord::new();
            let mut prev_ts = i64::MIN;
            loop {
                match reader.read_record(&mut record).await {
                    Ok(false) => break, // EOF
                    Err(e) => {
                        eprintln!("CsvSource error: {e}");
                        return;
                    }
                    Ok(true) => {}
                }
                let ts = match parse_timestamp(&record[time_idx]) {
                    Ok(t) => t + self.timestamp_offset_ns,
                    Err(e) => {
                        eprintln!("CsvSource error: {e}");
                        return;
                    }
                };
                assert!(
                    ts >= prev_ts,
                    "CsvSource: timestamps not sorted in {}",
                    self.path,
                );
                prev_ts = ts;

                let values = match parse_values(&record, &value_indices) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("CsvSource error: {e}");
                        return;
                    }
                };
                let arr = Array::from_vec(&[num_columns], values);
                if hist_tx.send((ts, arr)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, Array::zeros(&[num_columns]))
    }

    fn write(payload: Array<f64>, output: &mut Array<f64>, _timestamp: i64) -> bool {
        output.assign(payload.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_date() {
        let ts = parse_timestamp("2024-01-15").unwrap();
        // 2024-01-15 00:00:00 UTC
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap();
        assert_eq!(ts, expected);
    }

    #[test]
    fn parse_datetime() {
        // datetime → truncated to date
        let ts = parse_timestamp("2024-01-15 09:30:00").unwrap();
        let expected = chrono::NaiveDate::from_ymd_opt(2024, 1, 15)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_nanos_opt()
            .unwrap();
        assert_eq!(ts, expected);
    }
}
