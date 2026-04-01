//! Historical-only source that reads a CSV file.

use chrono::NaiveDate;
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
}

impl CsvSource {
    /// Create a new CSV source.
    ///
    /// * `path` — filesystem path to the CSV file.
    /// * `time_column` — header name of the date/datetime column
    ///   (parsed as `YYYY-MM-DD`).
    /// * `value_columns` — header names of columns to include as values,
    ///   in order.  Each is parsed as `f64`.
    pub fn new(path: String, time_column: String, value_columns: Vec<String>) -> Self {
        Self {
            path,
            time_column,
            value_columns,
        }
    }
}

/// A parsed row: timestamp (nanos since epoch) + flat f64 values.
struct Row {
    timestamp: i64,
    values: Vec<f64>,
}

/// Parse a date string (`YYYY-MM-DD` or `YYYY-MM-DD HH:MM:SS`) to
/// nanoseconds since the UNIX epoch (midnight UTC).
fn parse_timestamp(s: &str) -> Result<i64, String> {
    // Try date-only first, then datetime.
    let date = if let Ok(d) = NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d") {
        d
    } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(s.trim(), "%Y-%m-%d %H:%M:%S") {
        dt.date()
    } else {
        return Err(format!("cannot parse date: {s:?}"));
    };
    let nanos = date
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp_nanos_opt()
        .ok_or_else(|| format!("timestamp overflow for {s:?}"))?;
    Ok(nanos)
}

/// Read and parse the CSV, returning sorted rows.
fn read_csv(path: &str, time_column: &str, value_columns: &[String]) -> Result<Vec<Row>, String> {
    let mut reader =
        csv::Reader::from_path(path).map_err(|e| format!("cannot open {path}: {e}"))?;

    // Resolve column indices from headers.
    let headers = reader.headers().map_err(|e| e.to_string())?.clone();
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

    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let ts = parse_timestamp(&record[time_idx])?;
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
        rows.push(Row {
            timestamp: ts,
            values,
        });
    }

    // Sort by timestamp (stable, preserves row order for ties).
    rows.sort_by_key(|r| r.timestamp);
    Ok(rows)
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
        let num_columnss = self.value_columns.len();
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            let rows = match read_csv(&self.path, &self.time_column, &self.value_columns) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("CsvSource error: {e}");
                    return;
                }
            };
            for row in rows {
                let arr = Array::from_vec(&[num_columnss], row.values);
                if hist_tx.send((row.timestamp, arr)).await.is_err() {
                    break;
                }
            }
        });

        (hist_rx, live_rx, Array::zeros(&[num_columnss]))
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
