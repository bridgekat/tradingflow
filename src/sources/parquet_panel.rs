//! Cross-sectional panel source over a long-format Parquet table.
//!
//! [`ParquetPanelSource`] reads one **long** table — `(date, symbol, <value
//! columns…>)`, sorted by `(date, symbol)`, as written by
//! `a-shares-crawler --export-long parquet` — and emits one **wide
//! cross-section** per distinct `date`: an `Array<f64>` of shape `[N, K]`, where
//! `N` is the universe size (the `symbols` list, fixed order) and `K` is the
//! number of emitted value columns. Row `i` is symbol `symbols[i]`'s values for
//! that date; symbols **absent** that date are `NaN`.
//!
//! This replaces the per-symbol [`CsvSource`](super::CsvSource) fan-in with a
//! single sequential, date-ordered scan. Downstream, a per-stock pipeline is
//! recovered by `Select`ing one row of the panel:
//! `Select::new(vec![i], 0, true)` yields symbol `i`'s `[K]` vector, a drop-in
//! for the old per-symbol source output.
//!
//! # Semantics — pure StackSync; the carry lives downstream
//!
//! Each emitted cross-section reflects **only that date's rows** (absent symbols
//! are `NaN`); there is no carry-forward and no window-start seeding. This is the
//! event-driven behaviour of the old per-symbol sources: a source ticks only on
//! its own dates, and any "carry the last value" / "NaN-fill" is the job of the
//! downstream [`Stack`](crate::flow::Stack) / [`StackSync`](crate::flow::StackSync)
//! operators — not the source. With `with_time_range`, rows before `start` are
//! simply skipped (no last-value-before-`start` is carried in).
//!
//! # Irregular kinds & message-passing operators
//!
//! For daily prices every symbol ticks on (almost) every date, so the panel is
//! dense. For **irregular** kinds (dividends, financial reports) the panel emits
//! at the *union* of all symbols' event dates — which need not include every
//! trading day. A per-stock `Select` therefore still fires on that union cadence
//! with `NaN` where the stock had no row; a [`Filter`](crate::flow::Filter) that
//! drops the all-`NaN` ("no data") rows recovers that stock's true event stream,
//! so message-passing operators (e.g. [`ForwardAdjust`](crate::flow::ForwardAdjust))
//! see each real event exactly once — reproducing the per-symbol stream.
//!
//! # Timestamps
//!
//! `date` is a Parquet `date32` (days since 1970-01-01). The event timestamp is
//! the day's UTC-midnight instant via [`hifitime`] — identical to
//! [`CsvSource`](super::CsvSource)'s default parse, so a panel→select→filter
//! stream is timestamp-identical to the original per-symbol source.
//!
//! Requires a tokio runtime when added to a scenario (the Parquet scan runs on a
//! `spawn_blocking` task feeding the historical channel with back-pressure).

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::mpsc;

use arrow::array::{Array as _, ArrayRef, Date32Array, DictionaryArray, Int32Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int32Type};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use hifitime::{Duration as HfDuration, Epoch};

use crate::{Array, Instant, Source};

/// Historical-only source that pivots a long-format Parquet table into wide
/// `[N, K]` cross-sections, one per distinct `date`. See the module docs.
#[derive(Clone)]
pub struct ParquetPanelSource {
    path: String,
    value_columns: Vec<String>,
    symbols: Vec<String>,
    date_column: String,
    symbol_column: String,
    start: Option<Instant>,
    end: Option<Instant>,
    /// Shared row-progress counter (set by the engine); bumped per scanned row.
    progress: Option<Arc<AtomicU64>>,
}

impl ParquetPanelSource {
    /// Create a panel source.
    ///
    /// * `path` — the long-format Parquet file.
    /// * `value_columns` — value columns to emit, in order (each cast to `f64`).
    /// * `symbols` — the universe, in row order; each emitted cross-section has
    ///   one row per symbol. Symbols in the file but not in this list are ignored.
    pub fn new(path: impl Into<String>, value_columns: Vec<String>, symbols: Vec<String>) -> Self {
        Self {
            path: path.into(),
            value_columns,
            symbols,
            date_column: "date".into(),
            symbol_column: "symbol".into(),
            start: None,
            end: None,
            progress: None,
        }
    }

    /// Restrict emitted cross-sections to dates in `[start, end]` (inclusive).
    /// Rows before `start` are skipped (not carried in).
    pub fn with_time_range(mut self, start: Option<Instant>, end: Option<Instant>) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Override the `date` / `symbol` column names (defaults `"date"` / `"symbol"`).
    pub fn with_columns(mut self, date_column: impl Into<String>, symbol_column: impl Into<String>) -> Self {
        self.date_column = date_column.into();
        self.symbol_column = symbol_column.into();
        self
    }

    /// Emitted element shape, `[N, value_columns]`. Pass `Array::zeros(&out_shape())`
    /// as the `initial` value to [`Scenario::add_source`](crate::flow::Scenario::add_source).
    pub fn out_shape(&self) -> Vec<usize> {
        vec![self.symbols.len(), self.value_columns.len()]
    }
}

/// `date32` days since 1970-01-01 → the day's UTC-midnight [`Epoch`] (hifitime).
fn epoch_from_days(days: i32) -> Epoch {
    Epoch::from_unix_duration(HfDuration::from_truncated_nanoseconds(
        days as i64 * 86_400 * 1_000_000_000,
    ))
}

/// `date32` days → event [`Instant`] (UTC midnight → TAI), matching `CsvSource`.
pub(crate) fn instant_from_days(days: i32) -> Instant {
    Instant::from_hifitime_epoch(epoch_from_days(days))
}

/// `(year, day_of_year)` (1-based) for a report date, via the **same** hifitime
/// path as [`FinancialReportSource`](super::stocks::FinancialReportSource)
/// (`Epoch::year_days_of_year() + 1`), so [`Annualize`](crate::flow::Annualize)
/// matches it bit-for-bit. Used by [`ReportPanelSource`](super::ReportPanelSource).
pub(crate) fn report_year_and_doy(days: i32) -> (f64, f64) {
    let (year, day_of_year) = epoch_from_days(days).year_days_of_year();
    (year as f64, day_of_year + 1.0)
}

impl Source for ParquetPanelSource {
    type Event = Array<f64>;
    type Output = Array<f64>;

    fn estimated_event_count(&self) -> Option<usize> {
        // Progress is measured in **long-table rows scanned**. The scan starts at
        // the file head and breaks once past `end`, so it reads exactly the rows
        // with `date <= end` — which is also what it bumps the counter by. With
        // no end it reads the whole file (taken cheaply from parquet metadata).
        // On any read error fall back to `Some(0)` so the aggregate stays usable.
        Some(
            match self.end {
                Some(end) => count_rows_through_end(&self.path, &self.date_column, end),
                None => parquet_num_rows(&self.path),
            }
            .unwrap_or(0),
        )
    }

    fn install_progress_counter(&mut self, counter: Arc<AtomicU64>) {
        self.progress = Some(counter);
    }

    fn init(
        &self,
        _timestamp: Instant,
    ) -> (
        mpsc::Receiver<(Instant, Array<f64>)>,
        mpsc::Receiver<(Instant, Array<f64>)>,
        Array<f64>,
    ) {
        let (hist_tx, hist_rx) = mpsc::channel(64);
        let (_, live_rx) = mpsc::channel(1);
        let cfg = self.clone();
        let out_shape = self.out_shape();

        tokio::task::spawn_blocking(move || {
            if let Err(e) = read_panel(&cfg, &hist_tx) {
                eprintln!("ParquetPanelSource error ({}): {e}", cfg.path);
            }
        });

        (hist_rx, live_rx, Array::zeros(&out_shape))
    }

    fn write(payload: Array<f64>, output: &mut Array<f64>, _timestamp: Instant) -> bool {
        output.assign(payload.as_slice());
        true
    }
}

/// Sequential scan: read row groups in `(date, symbol)` order and emit one
/// `[N, K]` cross-section per distinct date (NaN where a symbol has no row).
fn read_panel(
    cfg: &ParquetPanelSource,
    hist_tx: &mpsc::Sender<(Instant, Array<f64>)>,
) -> Result<(), String> {
    let n = cfg.symbols.len();
    let k = cfg.value_columns.len();
    let sym_index: HashMap<&str, usize> =
        cfg.symbols.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let file = File::open(&cfg.path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;

    // Project only the columns we read so the wide statement tables don't decode
    // their ~180 unused line-item columns.
    let schema = builder.schema();
    let mut needed: Vec<&str> = vec![cfg.date_column.as_str(), cfg.symbol_column.as_str()];
    needed.extend(cfg.value_columns.iter().map(|s| s.as_str()));
    let leaf_indices: Vec<usize> = needed
        .iter()
        .map(|name| schema.index_of(name).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), leaf_indices);
    let reader = builder.with_projection(mask).build().map_err(|e| e.to_string())?;

    let (start, end) = (cfg.start, cfg.end);
    let mut buf = vec![f64::NAN; n * k];
    let mut current: Option<i32> = None;
    let mut written: Vec<usize> = Vec::new(); // rows set for `current` (sparse reset)
    let send = |ts: Instant, buf: &[f64]| -> bool {
        hist_tx.blocking_send((ts, Array::from_vec(&[n, k], buf.to_vec()))).is_ok()
    };

    'outer: for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;

        let date_col = batch
            .column_by_name(&cfg.date_column)
            .ok_or_else(|| format!("missing date column {:?}", cfg.date_column))?;
        let dates = date_col
            .as_any()
            .downcast_ref::<Date32Array>()
            .ok_or_else(|| format!("date column {:?} is not date32", cfg.date_column))?;

        let sym_col = batch
            .column_by_name(&cfg.symbol_column)
            .ok_or_else(|| format!("missing symbol column {:?}", cfg.symbol_column))?;
        let row_uni = resolve_symbols(sym_col, &sym_index)?;

        // Cast each value column to f64 once per batch.
        let val_refs: Vec<ArrayRef> = cfg
            .value_columns
            .iter()
            .map(|c| {
                let col = batch
                    .column_by_name(c)
                    .ok_or_else(|| format!("missing value column {c:?}"))?;
                cast(col.as_ref(), &DataType::Float64).map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        let vals: Vec<&arrow::array::Float64Array> = val_refs
            .iter()
            .map(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap())
            .collect();

        let mut seen: u64 = 0; // long-table rows scanned in this batch (progress unit)
        for row in 0..batch.num_rows() {
            let d = dates.value(row);
            let ts = instant_from_days(d);
            // (end-check before start-check is equivalent since `start <= end`,
            // and lets us count every scanned row in `[.., end]` — including the
            // pre-`start` rows the scan reads then skips — toward progress.)
            if end.is_some_and(|e| ts > e) {
                if let Some(c) = cfg.progress.as_ref() {
                    c.fetch_add(seen, Ordering::Relaxed);
                }
                break 'outer;
            }
            seen += 1;
            if start.is_some_and(|s| ts < s) {
                continue; // before the window: skip (no carry-in)
            }
            if current != Some(d) {
                if let Some(cd) = current {
                    if !send(instant_from_days(cd), &buf) {
                        return Ok(());
                    }
                    // Sparse reset: clear only the rows written for the prev date.
                    for &idx in &written {
                        buf[idx * k..idx * k + k].fill(f64::NAN);
                    }
                    written.clear();
                }
                current = Some(d);
            }
            let Some(ui) = row_uni[row] else { continue };
            let base = ui * k;
            for (vi, va) in vals.iter().enumerate() {
                buf[base + vi] = if va.is_null(row) { f64::NAN } else { va.value(row) };
            }
            written.push(ui);
        }
        if let Some(c) = cfg.progress.as_ref() {
            c.fetch_add(seen, Ordering::Relaxed);
        }
    }

    if let Some(cd) = current {
        send(instant_from_days(cd), &buf);
    }
    Ok(())
}

/// Total row count from the parquet footer metadata — O(1), no row decode.
/// The number of rows a full scan (no `end` bound) reads.
pub(crate) fn parquet_num_rows(path: &str) -> Result<usize, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    Ok(builder.metadata().file_metadata().num_rows().max(0) as usize)
}

/// Count rows with `date <= end` by scanning **only** the date column (the table
/// is `(date, symbol)`-sorted, so the scan can stop at the first row past `end`;
/// the column is tiny once zstd-compressed). This is exactly how many long-table
/// rows [`read_panel`] reads with an `end` bound — the progress-estimate unit.
pub(crate) fn count_rows_through_end(
    path: &str,
    date_column: &str,
    end: Instant,
) -> Result<usize, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let leaf = builder.schema().index_of(date_column).map_err(|e| e.to_string())?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), vec![leaf]);
    let reader = builder.with_projection(mask).build().map_err(|e| e.to_string())?;

    let mut count = 0usize;
    'outer: for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;
        let dates = batch
            .column_by_name(date_column)
            .ok_or_else(|| format!("missing date column {date_column:?}"))?
            .as_any()
            .downcast_ref::<Date32Array>()
            .ok_or_else(|| format!("date column {date_column:?} is not date32"))?;
        for row in 0..batch.num_rows() {
            if instant_from_days(dates.value(row)) > end {
                break 'outer;
            }
            count += 1;
        }
    }
    Ok(count)
}

/// Map each batch row to its universe index (or `None` if the symbol is not in
/// the universe / is null). Fast path for dictionary-encoded symbol columns.
pub(crate) fn resolve_symbols(
    sym_col: &ArrayRef,
    sym_index: &HashMap<&str, usize>,
) -> Result<Vec<Option<usize>>, String> {
    if let Some(dict) = sym_col.as_any().downcast_ref::<DictionaryArray<Int32Type>>() {
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("dictionary values are not utf8")?;
        // Dictionary entry -> universe index (computed once; dictionary is small).
        let entry_uni: Vec<Option<usize>> =
            (0..values.len()).map(|key| sym_index.get(values.value(key)).copied()).collect();
        let keys: &Int32Array = dict.keys();
        Ok((0..keys.len())
            .map(|row| {
                if keys.is_null(row) {
                    None
                } else {
                    entry_uni[keys.value(row) as usize]
                }
            })
            .collect())
    } else {
        // Fallback: materialize to utf8 and hash per row.
        let utf8 = cast(sym_col.as_ref(), &DataType::Utf8).map_err(|e| e.to_string())?;
        let sa = utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("symbol column is not string-castable")?;
        Ok((0..sa.len())
            .map(|row| if sa.is_null(row) { None } else { sym_index.get(sa.value(row)).copied() })
            .collect())
    }
}
