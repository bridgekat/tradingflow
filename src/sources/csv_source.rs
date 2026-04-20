//! Historical-only source that reads a CSV file asynchronously.

use csv_async::StringRecord;
use hifitime::Epoch;
use tokio::sync::mpsc;

use crate::{Duration, Instant};
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
    timestamp_offset: Duration,
    /// If `true` (default), date strings are interpreted as UTC wall-clock
    /// instants and converted to TAI via the IERS leap-second table.  If
    /// `false`, they are interpreted directly as TAI wall-clock (no leap
    /// seconds involved).
    is_utc: bool,
    /// Offset of the date-string timezone from the reference timescale.
    /// E.g. `Duration::from_hours(8)` for Asia/Shanghai when `is_utc` is
    /// `true`.  The parsed local wall-clock is shifted by subtracting
    /// `tz_offset` before leap-second conversion.
    tz_offset: Duration,
    /// Optional inclusive start bound.  Rows before this timestamp are dropped.
    start: Option<Instant>,
    /// Optional inclusive end bound.  Rows after this timestamp are dropped.
    end: Option<Instant>,
}

impl CsvSource {
    /// Create a new CSV source.
    ///
    /// * `path` — filesystem path to the CSV file.
    /// * `time_column` — header name of the date/datetime column
    ///   (parsed as `YYYY-MM-DD`).
    /// * `value_columns` — header names of columns to include as values,
    ///   in order.  Each is parsed as `f64`.
    /// * `timestamp_offset` — constant offset added to every parsed
    ///   timestamp before it is used as the event timestamp.  Useful when
    ///   the CSV contains low-precision timestamps (e.g. dates) that would
    ///   otherwise cause forward-looking bias against higher-precision
    ///   sources.
    ///
    /// The date strings are interpreted as **UTC midnight** by default.
    /// Use [`with_timescale`](Self::with_timescale) to change the
    /// interpretation (TAI) or the wall-clock timezone offset.
    pub fn new(
        path: String,
        time_column: String,
        value_columns: Vec<String>,
        timestamp_offset: Duration,
    ) -> Self {
        Self {
            path,
            time_column,
            value_columns,
            timestamp_offset,
            is_utc: true,
            tz_offset: Duration::ZERO,
            start: None,
            end: None,
        }
    }

    /// Set the interpretation of the date strings in the CSV.
    ///
    /// * `is_utc` — if `true` (default), date strings are interpreted as
    ///   UTC wall-clock instants; conversion to TAI applies the IERS
    ///   leap-second offset.  If `false`, they are interpreted as TAI
    ///   wall-clock directly (no leap-second math).
    /// * `tz_offset` — offset of the local wall-clock timezone from the
    ///   reference timescale (UTC or TAI as selected by `is_utc`).  For
    ///   example, `Duration::from_hours(8)` for Asia/Shanghai.
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

/// Estimate the number of data rows in a CSV file from a small synchronous
/// prefix read.  Samples at most `SAMPLE_BYTES` to compute an average line
/// length after the header, then extrapolates against the full file size.
///
/// For CSV files dominated by fixed-width numeric columns the resulting
/// estimate is usually within a few percent.  Header-only files return
/// `Some(0)`.  Returns `None` only if the file cannot be opened or read.
pub(crate) fn estimate_csv_rows(path: &str) -> Option<usize> {
    const SAMPLE_BYTES: usize = 16 * 1024;
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let file_size = f.metadata().ok()?.len();
    if file_size == 0 {
        return Some(0);
    }
    let mut buf = vec![0u8; SAMPLE_BYTES.min(file_size as usize)];
    let read = f.read(&mut buf).ok()?;
    buf.truncate(read);
    // Find the end of the header line.  No newline → header-only (no body).
    let Some(nl_pos) = buf.iter().position(|&b| b == b'\n') else {
        return Some(0);
    };
    let header_end = nl_pos + 1;
    let body = &buf[header_end..];
    let line_count = body.iter().filter(|&&b| b == b'\n').count();
    if line_count == 0 {
        // Header line + at most one partial data row in the sample.  Treat
        // as header-only when the whole file fits in the sample, otherwise
        // approximate from the partial data row's length.
        if file_size as usize <= read {
            return Some(if body.is_empty() { 0 } else { 1 });
        }
        let avg_line_bytes = body.len().max(1) as f64;
        let remaining_bytes = file_size.saturating_sub(header_end as u64) as f64;
        return Some((remaining_bytes / avg_line_bytes).round() as usize);
    }
    let avg_line_bytes = body.len() as f64 / line_count as f64;
    let remaining_bytes = file_size.saturating_sub(header_end as u64) as f64;
    Some((remaining_bytes / avg_line_bytes).round() as usize)
}

/// Scale a row estimate by the fraction of the file's time span that
/// overlaps `[start, end]`, assuming rows are uniformly distributed in
/// time.
///
/// The file's time span is read from the first and last data rows via
/// [`read_boundary_timestamps`], which does one seek + two small reads.
/// When neither `start` nor `end` is set, or the boundary reads fail, the
/// input `total` is returned unchanged.
pub(crate) fn scale_rows_to_range(
    total: usize,
    path: &str,
    time_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
    is_utc: bool,
    tz_offset: Duration,
    timestamp_offset: Duration,
) -> usize {
    if total == 0 || (start.is_none() && end.is_none()) {
        return total;
    }
    let Some((file_start, file_end)) =
        read_boundary_timestamps(path, time_column, is_utc, tz_offset, timestamp_offset)
    else {
        return total;
    };
    if file_end <= file_start {
        return total;
    }
    let q_start = start.unwrap_or(Instant::MIN).max(file_start);
    let q_end = end.unwrap_or(Instant::MAX).min(file_end);
    if q_end <= q_start {
        return 0;
    }
    let span = (file_end - file_start).as_nanos() as f64;
    let overlap = (q_end - q_start).as_nanos() as f64;
    ((total as f64) * (overlap / span).clamp(0.0, 1.0)).round() as usize
}

/// Read the first and last data rows' time-column values from a CSV file.
///
/// Uses one prefix read for the header + first row and one suffix seek for
/// the tail.  Assumes simple CSVs with no quoted newlines and the time
/// column containing a leading `YYYY-MM-DD` date with no embedded commas.
/// Returns `None` if the file has fewer than two data rows, the column
/// cannot be resolved, or a date fails to parse.
fn read_boundary_timestamps(
    path: &str,
    time_column: &str,
    is_utc: bool,
    tz_offset: Duration,
    timestamp_offset: Duration,
) -> Option<(Instant, Instant)> {
    const BOUNDARY_SAMPLE_BYTES: usize = 4 * 1024;
    use std::io::{Read, Seek, SeekFrom};

    let mut f = std::fs::File::open(path).ok()?;
    let file_size = f.metadata().ok()?.len() as usize;
    if file_size == 0 {
        return None;
    }

    // Prefix: header + first data row.
    let prefix_len = BOUNDARY_SAMPLE_BYTES.min(file_size);
    let mut prefix = vec![0u8; prefix_len];
    f.read_exact(&mut prefix).ok()?;
    let prefix_str = std::str::from_utf8(&prefix).ok()?;
    let mut lines = prefix_str.split('\n');
    let header = lines.next()?;
    let first_data_line = lines.next()?.trim_end_matches('\r');
    if first_data_line.is_empty() {
        return None;
    }
    let time_idx = header
        .trim_end_matches('\r')
        .split(',')
        .position(|h| h.trim() == time_column)?;
    let first_ts =
        parse_ts_from_line(first_data_line, time_idx, is_utc, tz_offset, timestamp_offset)?;

    // Suffix: find the last complete data line.
    let tail_len = BOUNDARY_SAMPLE_BYTES.min(file_size);
    let tail_bytes: Vec<u8> = if file_size <= prefix_len {
        prefix
    } else {
        f.seek(SeekFrom::End(-(tail_len as i64))).ok()?;
        let mut tail = vec![0u8; tail_len];
        f.read_exact(&mut tail).ok()?;
        tail
    };
    let tail_str = std::str::from_utf8(&tail_bytes).ok()?;
    let trimmed = tail_str.trim_end_matches(|c: char| c == '\n' || c == '\r');
    // Find the start of the last line: after the last '\n' if any, else the
    // beginning of the tail (valid only if the whole file fit in the buffer).
    let last_line_start = match trimmed.rfind('\n') {
        Some(i) => i + 1,
        None if file_size <= tail_len => 0,
        None => return None,
    };
    let last_line = trimmed[last_line_start..].trim_end_matches('\r');
    if last_line.is_empty() {
        return None;
    }
    let last_ts =
        parse_ts_from_line(last_line, time_idx, is_utc, tz_offset, timestamp_offset)?;

    Some((first_ts, last_ts))
}

/// Parse the `time_idx`-th comma-separated field of `line` as a
/// `YYYY-MM-DD` date and convert it to an [`Instant`] under the given
/// timescale interpretation.
fn parse_ts_from_line(
    line: &str,
    time_idx: usize,
    is_utc: bool,
    tz_offset: Duration,
    timestamp_offset: Duration,
) -> Option<Instant> {
    let field = line.split(',').nth(time_idx)?.trim();
    let epoch = parse_gregorian_date(field).ok()?;
    Some(epoch_to_instant(epoch, is_utc, tz_offset) + timestamp_offset)
}

/// Parse the leading `YYYY-MM-DD` of a string into a Gregorian `Epoch`
/// (at UTC midnight, as if the date were in UTC).  Any time component
/// after the leading 10-char date is ignored.  The resulting Epoch is
/// the one hifitime returns from `maybe_from_gregorian_utc`; callers
/// re-anchor it as needed for UTC vs TAI interpretation.
pub(crate) fn parse_gregorian_date(s: &str) -> Result<Epoch, String> {
    let s = s.trim();
    let date = s.get(..10).ok_or_else(|| format!("cannot parse date: {s:?}"))?;
    let mut parts = date.split('-');
    let year: i32 = parts.next().and_then(|p| p.parse().ok())
        .ok_or_else(|| format!("cannot parse date: {s:?}"))?;
    let month: u8 = parts.next().and_then(|p| p.parse().ok())
        .ok_or_else(|| format!("cannot parse date: {s:?}"))?;
    let day: u8 = parts.next().and_then(|p| p.parse().ok())
        .ok_or_else(|| format!("cannot parse date: {s:?}"))?;
    if parts.next().is_some() {
        return Err(format!("cannot parse date: {s:?}"));
    }
    Epoch::maybe_from_gregorian_utc(year, month, day, 0, 0, 0, 0)
        .map_err(|e| format!("invalid date {s:?}: {e}"))
}

/// Convert a Gregorian-parsed `Epoch` to an [`Instant`] under the
/// source's timescale interpretation, then shift by `tz_offset`.
///
/// `is_utc = true`: the date labels a UTC wall-clock instant; the TAI
/// `Instant` is that UTC instant viewed on the TAI timeline (leap-second
/// offset applied by hifitime).
///
/// `is_utc = false`: the date labels a TAI wall-clock instant; every
/// calendar day is 86 400 SI seconds (no leap-second math).  The TAI
/// `Instant` is the integer `(days_since_1970 * 86400 + seconds_of_day)
/// * 1e9`, which equals the UNIX-ns of the same date reinterpreted on
/// the TAI timeline.
pub(crate) fn epoch_to_instant(epoch: Epoch, is_utc: bool, tz_offset: Duration) -> Instant {
    let anchored = if is_utc {
        Instant::from_hifitime_epoch(epoch)
    } else {
        Instant::from_nanos(Instant::from_hifitime_epoch(epoch).to_utc_nanos())
    };
    anchored - tz_offset
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

    fn estimated_event_count(&self) -> Option<usize> {
        let total = estimate_csv_rows(&self.path)?;
        Some(scale_rows_to_range(
            total,
            &self.path,
            &self.time_column,
            self.start,
            self.end,
            self.is_utc,
            self.tz_offset,
            self.timestamp_offset,
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

            let start = self.start;
            let end = self.end;
            let is_utc = self.is_utc;
            let tz_offset = self.tz_offset;

            // When `start` is set, track the last row before the window
            // so it can be emitted as the initial value at `start`.
            let mut last_before_start: Option<Vec<f64>> = None;
            let mut entered_window = start.is_none();

            let mut record = StringRecord::new();
            let mut prev_ts = Instant::MIN;
            loop {
                match reader.read_record(&mut record).await {
                    Ok(false) => break, // EOF
                    Err(e) => {
                        eprintln!("CsvSource error: {e}");
                        return;
                    }
                    Ok(true) => {}
                }
                let ts = match parse_gregorian_date(&record[time_idx]) {
                    Ok(epoch) => epoch_to_instant(epoch, is_utc, tz_offset) + self.timestamp_offset,
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

                // Before the start of the window: remember the last row.
                if let Some(s) = start {
                    if ts < s {
                        last_before_start = Some(values);
                        continue;
                    }
                }

                // First row at or after start: emit the carried-over
                // initial value (if any) at the start timestamp.
                if !entered_window {
                    entered_window = true;
                    if let Some(init_vals) = last_before_start.take() {
                        let s = start.unwrap();
                        let arr = Array::from_vec(&[num_columns], init_vals);
                        if hist_tx.send((s, arr)).await.is_err() {
                            break;
                        }
                    }
                }

                if let Some(e) = end {
                    if ts > e {
                        break;
                    }
                }

                let arr = Array::from_vec(&[num_columns], values);
                if hist_tx.send((ts, arr)).await.is_err() {
                    break;
                }
            }

            // If the file had only pre-start rows, emit the last one.
            if !entered_window {
                if let Some(init_vals) = last_before_start.take() {
                    if let Some(s) = start {
                        let arr = Array::from_vec(&[num_columns], init_vals);
                        let _ = hist_tx.send((s, arr)).await;
                    }
                }
            }
        });

        (hist_rx, live_rx, Array::zeros(&[num_columns]))
    }

    fn write(payload: Array<f64>, output: &mut Array<f64>, _timestamp: Instant) -> bool {
        output.assign(payload.as_slice());
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc_midnight(y: i32, m: u8, d: u8) -> Instant {
        Instant::from_hifitime_epoch(
            Epoch::maybe_from_gregorian_utc(y, m, d, 0, 0, 0, 0).unwrap(),
        )
    }

    fn parse_utc(s: &str) -> Instant {
        epoch_to_instant(parse_gregorian_date(s).unwrap(), true, Duration::ZERO)
    }

    #[test]
    fn parse_date_utc_default() {
        assert_eq!(parse_utc("2024-01-15"), utc_midnight(2024, 1, 15));
    }

    #[test]
    fn parse_datetime_utc_default() {
        // datetime → truncated to date
        assert_eq!(parse_utc("2024-01-15 09:30:00"), utc_midnight(2024, 1, 15));
    }

    #[test]
    fn parse_date_tai_interpretation() {
        // Under is_utc=false, "2024-01-15" labels TAI 2024-01-15 00:00:00
        // directly — no leap-second offset.
        let tai = epoch_to_instant(parse_gregorian_date("2024-01-15").unwrap(), false, Duration::ZERO);
        let utc = parse_utc("2024-01-15");
        // TAI interpretation is 37 s earlier on the TAI timeline than the
        // UTC interpretation of the same string (because UTC midnight was
        // 37 s later in TAI than TAI midnight).
        assert_eq!((utc - tai).as_nanos(), 37 * 1_000_000_000);
    }

    #[test]
    fn parse_date_with_tz_offset() {
        // "2024-01-15" in UTC+8 means UTC 2024-01-14 16:00:00.
        let shanghai = epoch_to_instant(
            parse_gregorian_date("2024-01-15").unwrap(),
            true,
            Duration::from_hours(8),
        );
        let utc = parse_utc("2024-01-15");
        assert_eq!((utc - shanghai).as_nanos(), 8 * 3600 * 1_000_000_000);
    }

    #[test]
    fn estimate_rows_approximate() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "date,open,close").unwrap();
        for i in 0..1000 {
            writeln!(
                f,
                "2024-01-{:02},{:.4},{:.4}",
                (i % 28) + 1,
                100.0 + i as f64,
                101.0 + i as f64,
            )
            .unwrap();
        }
        f.flush().unwrap();
        let est = estimate_csv_rows(f.path().to_str().unwrap()).unwrap();
        // Heuristic should be within 5% for uniform rows.
        assert!(
            (950..=1050).contains(&est),
            "expected ~1000, got {est}"
        );
    }

    #[test]
    fn estimate_rows_exact_when_file_fits_in_sample() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "date,val").unwrap();
        for i in 0..10 {
            writeln!(f, "2024-01-{:02},{i}.0", i + 1).unwrap();
        }
        f.flush().unwrap();
        let est = estimate_csv_rows(f.path().to_str().unwrap()).unwrap();
        assert_eq!(est, 10);
    }

    #[test]
    fn estimate_rows_header_only_returns_zero() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "date,val").unwrap();
        f.flush().unwrap();
        let est = estimate_csv_rows(f.path().to_str().unwrap()).unwrap();
        assert_eq!(est, 0);
    }

    #[test]
    fn estimate_rows_empty_file_returns_zero() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let est = estimate_csv_rows(f.path().to_str().unwrap()).unwrap();
        assert_eq!(est, 0);
    }

    /// Helper: write `days` daily rows starting at 2020-01-01, return path.
    fn write_daily_csv(f: &mut tempfile::NamedTempFile, days: u32) {
        use std::io::Write;
        writeln!(f, "date,val").unwrap();
        let mut year = 2020;
        let mut month = 1;
        let mut day = 1;
        for i in 0..days {
            writeln!(f, "{year:04}-{month:02}-{day:02},{i}").unwrap();
            day += 1;
            if day > 28 {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }
        f.flush().unwrap();
    }

    #[test]
    fn scale_rows_to_range_halves_when_half_overlap() {
        // 1000 daily rows spanning ~3.5 years; query the latter half.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write_daily_csv(&mut f, 1000);
        let path = f.path().to_str().unwrap();
        let total = estimate_csv_rows(path).unwrap();
        // Midpoint in UTC TAI ns via parse_utc.
        let file_start = parse_utc("2020-01-01");
        let file_end = parse_utc("2022-12-28"); // not exact but close enough
        let mid = Instant::from_nanos((file_start.as_nanos() + file_end.as_nanos()) / 2);
        let scaled = scale_rows_to_range(
            total,
            path,
            "date",
            Some(mid),
            None,
            true,
            Duration::ZERO,
            Duration::ZERO,
        );
        // Expect ~half the rows, allow ±10% for boundary-row parsing noise.
        let expected = total / 2;
        let lo = (expected as f64 * 0.90) as usize;
        let hi = (expected as f64 * 1.10) as usize + 2;
        assert!(
            (lo..=hi).contains(&scaled),
            "expected ~{expected}, got {scaled}"
        );
    }

    #[test]
    fn scale_rows_to_range_empty_window_returns_zero() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write_daily_csv(&mut f, 100);
        let path = f.path().to_str().unwrap();
        let total = estimate_csv_rows(path).unwrap();
        // Window entirely after the file.
        let scaled = scale_rows_to_range(
            total,
            path,
            "date",
            Some(parse_utc("2030-01-01")),
            Some(parse_utc("2031-01-01")),
            true,
            Duration::ZERO,
            Duration::ZERO,
        );
        assert_eq!(scaled, 0);
    }

    #[test]
    fn scale_rows_to_range_no_bounds_returns_total() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write_daily_csv(&mut f, 100);
        let path = f.path().to_str().unwrap();
        let total = estimate_csv_rows(path).unwrap();
        let scaled = scale_rows_to_range(
            total,
            path,
            "date",
            None,
            None,
            true,
            Duration::ZERO,
            Duration::ZERO,
        );
        assert_eq!(scaled, total);
    }
}
