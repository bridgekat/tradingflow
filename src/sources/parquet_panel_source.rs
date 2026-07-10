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
//! It reads the panel in a single sequential, date-ordered scan. Downstream, a
//! per-stock pipeline is recovered by `Select`ing one row of the panel:
//! `Select::new(vec![i], 0, true)` yields symbol `i`'s `[K]` vector.
//!
//! # Semantics — pure StackSync; the carry lives downstream
//!
//! Each emitted cross-section reflects **only that date's rows** (absent symbols
//! are `NaN`); there is no carry-forward and no window-start seeding. This is the
//! event-driven behaviour of the old per-symbol sources: a source ticks only on
//! its own dates, and any "carry the last value" / "NaN-fill" is the job of the
//! downstream [`Stack`](crate::operators::Stack) / [`StackSync`](crate::operators::StackSync)
//! operators — not the source. With `with_time_range`, rows before `start` are
//! simply skipped (no last-value-before-`start` is carried in).
//!
//! # Irregular kinds & message-passing operators
//!
//! For daily prices every symbol ticks on (almost) every date, so the panel is
//! dense. For **irregular** kinds (dividends, financial reports) the panel emits
//! at the *union* of all symbols' event dates — which need not include every
//! trading day. A per-stock `Select` therefore still fires on that union cadence
//! with `NaN` where the stock had no row; a [`Filter`](crate::operators::Filter) that
//! drops the all-`NaN` ("no data") rows recovers that stock's true event stream,
//! so message-passing operators (e.g. [`ForwardAdjust`](crate::operators::ForwardAdjust))
//! see each real event exactly once — reproducing the per-symbol stream.
//!
//! # Timestamps
//!
//! `date` is a Parquet `date32` (days since 1970-01-01). The event timestamp is
//! the day's UTC-midnight instant via [`hifitime`], so a panel→select→filter
//! stream is timestamp-aligned across kinds.
//!
//! Requires a tokio runtime when added to a scenario (the Parquet scan runs on a
//! `spawn_blocking` task feeding the historical channel with back-pressure).

use std::collections::HashMap;
use std::fs::File;

use futures::stream::Stream;
use tokio::sync::mpsc;

use flowgraph::ingest::{Event, EventSource};

use super::receiver_stream;

use arrow::array::{Array as _, ArrayRef, Date32Array, DictionaryArray, Int32Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int32Type};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::statistics::Statistics;

use hifitime::{Duration as HfDuration, Epoch};

use crate::{Array, Instant};

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

    /// Emitted element shape, `[N, value_columns]`.
    pub fn out_shape(&self) -> [usize; 2] {
        [self.symbols.len(), self.value_columns.len()]
    }
}

/// `date32` days since 1970-01-01 → the day's UTC-midnight [`Epoch`] (hifitime).
fn epoch_from_days(days: i32) -> Epoch {
    Epoch::from_unix_duration(HfDuration::from_truncated_nanoseconds(
        days as i64 * 86_400 * 1_000_000_000,
    ))
}

/// `date32` days → event [`Instant`] (UTC midnight → TAI).
pub(crate) fn instant_from_days(days: i32) -> Instant {
    Instant::from_utc_days(days as i64)
}

/// `(year, day_of_year)` (1-based) for a report date via hifitime
/// (`Epoch::year_days_of_year() + 1`), so [`Annualize`](crate::operators::Annualize)
/// consumes it directly. Used by [`ParquetFinancialReportPanelSource`](super::ParquetFinancialReportPanelSource).
pub(crate) fn report_year_and_doy(days: i32) -> (f64, f64) {
    let (year, day_of_year) = epoch_from_days(days).year_days_of_year();
    (year as f64, day_of_year + 1.0)
}

/// One narrow-table row's update to a wide `[N, K]` cross-section cell: symbol
/// `row`'s `K` values (NaN for null cells). The panel sources **batch a whole
/// tick's rows** into one channel item (`Vec<RowUpdate>`) — one message per
/// distinct date rather than per row, amortising the channel / event-loop cost —
/// and [`panel_write`] applies them per event (still the per-row state logic) to
/// reassemble the cross-section, returning the batch size so the run's event
/// count stays equal to the long-table row count.
pub struct RowUpdate {
    pub row: usize,
    pub vals: Box<[f64]>,
}

/// Per-source write state for the panel sources: the timestamp of the tick
/// being assembled and the rows it dirtied. When the timestamp **strictly
/// advances**, [`panel_write`] NaN-clears those rows first — reproducing the
/// per-tick "only this date's rows" cross-section (pure StackSync, no carry).
pub struct PanelState {
    last_ts: Option<Instant>,
    dirty: Vec<usize>,
}

impl Default for PanelState {
    fn default() -> Self {
        Self { last_ts: None, dirty: Vec::new() }
    }
}

/// Shared `Source::write` body for both panel sources: apply one tick's batch.
/// On a new tick (the batch's `ts` strictly past the last) clear the previous
/// tick's rows first, then write each row's `K` values **per event**. Every row
/// in a batch shares the tick's `ts` and the source's `K` (taken from the first
/// row, so it works for both 2-D `[N, K]` and squeezed `[N]` cells). Returns the
/// number of rows applied — the per-event count the run reports.
/// The all-NaN initial panel: the per-tick `write` only sets rows that have
/// an event, so an unwritten row must read as `NaN` (not `0.0`) for the
/// per-stock `Filter` to drop it.
pub(crate) fn nan_panel(shape: [usize; 2]) -> Array<f64, 2> {
    Array::from_vec(shape, vec![f64::NAN; shape.iter().product()])
}

pub(crate) fn panel_write(
    state: &mut PanelState,
    batch: Vec<RowUpdate>,
    output: &mut Array<f64, 2>,
    ts: Instant,
) -> usize {
    let Some(first) = batch.first() else { return 0 };
    let k = first.vals.len();
    let buf = output.as_mut_slice();
    if state.last_ts.is_some_and(|last| ts > last) {
        for &r in &state.dirty {
            buf[r * k..r * k + k].fill(f64::NAN);
        }
        state.dirty.clear();
    }
    state.last_ts = Some(ts);
    let n = batch.len();
    for ev in batch {
        let base = ev.row * k;
        buf[base..base + k].copy_from_slice(&ev.vals);
        state.dirty.push(ev.row);
    }
    n
}

impl EventSource<Instant> for ParquetPanelSource {
    type Event = Vec<RowUpdate>;
    type Output = Array<f64, 2>;
    type State = PanelState;

    fn estimated_event_count(&self) -> Option<usize> {
        // Progress is measured in **emitted long-table rows** (one event per
        // universe row in `[start, end]`). No time range → the whole file (O(1)
        // from the footer); otherwise count rows in the window. On any read
        // error fall back to `Some(0)` so the aggregate stays usable.
        Some(
            if self.start.is_none() && self.end.is_none() {
                parquet_num_rows(&self.path)
            } else {
                count_rows_in_range(&self.path, &self.date_column, self.start, self.end)
            }
            .unwrap_or(0),
        )
    }

    fn initial(&self) -> Array<f64, 2> {
        nan_panel(self.out_shape())
    }

    fn init(
        &self,
    ) -> (impl Stream<Item = Event<Instant, Vec<RowUpdate>>> + Send + 'static, PanelState) {
        // Each item is now a whole tick's rows, so a small buffer pipelines plenty
        // of ticks ahead while bounding the in-flight row memory.
        let (hist_tx, hist_rx) = mpsc::channel(16);
        let cfg = self.clone();

        tokio::task::spawn_blocking(move || {
            if let Err(e) = read_panel(&cfg, &hist_tx) {
                eprintln!("ParquetPanelSource error ({}): {e}", cfg.path);
            }
        });

        (receiver_stream(hist_rx), PanelState::default())
    }

    fn write(state: &mut PanelState, batch: Vec<RowUpdate>, output: &mut Array<f64, 2>, ts: Instant) -> usize {
        panel_write(state, batch, output, ts)
    }
}

/// Sequential scan: read row groups in `(date, symbol)` order and emit **one
/// batch per distinct date** — a `Vec<RowUpdate>` of that date's in-universe,
/// in-window rows, timestamped by the date. Buffering rows until the date
/// changes (and flushing the last date after the scan) keeps one channel message
/// per tick; [`panel_write`] then reassembles each date's `[N, K]` cross-section
/// and clears the previous date's rows when the timestamp advances.
fn read_panel(
    cfg: &ParquetPanelSource,
    hist_tx: &mpsc::Sender<(Instant, Vec<RowUpdate>)>,
) -> Result<(), String> {
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
    // Seek past row groups whose date range is entirely outside the window.
    let row_groups = date_row_groups_in_range(&builder, &cfg.date_column, cfg.start, cfg.end);
    let reader = builder
        .with_projection(mask)
        .with_row_groups(row_groups)
        .build()
        .map_err(|e| e.to_string())?;

    let (start, end) = (cfg.start, cfg.end);
    let mut cur_ts: Option<Instant> = None;
    let mut tick: Vec<RowUpdate> = Vec::new();

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

        for row in 0..batch.num_rows() {
            let ts = instant_from_days(dates.value(row));
            if end.is_some_and(|e| ts > e) {
                break 'outer; // sorted by date: nothing left in the window
            }
            if start.is_some_and(|s| ts < s) {
                continue; // before the window: skip (no carry-in)
            }
            let Some(ui) = row_uni[row] else { continue };
            // Date changed → ship the accumulated tick as one batch.
            if cur_ts != Some(ts) {
                if let Some(t) = cur_ts {
                    if hist_tx.blocking_send((t, std::mem::take(&mut tick))).is_err() {
                        return Ok(());
                    }
                }
                cur_ts = Some(ts);
            }
            let mut payload = vec![f64::NAN; k];
            for (vi, va) in vals.iter().enumerate() {
                if !va.is_null(row) {
                    payload[vi] = va.value(row);
                }
            }
            tick.push(RowUpdate { row: ui, vals: payload.into_boxed_slice() });
        }
    }
    // Flush the final tick (also the last in-window tick when we broke on `end`).
    if let Some(t) = cur_ts {
        if !tick.is_empty() {
            let _ = hist_tx.blocking_send((t, tick));
        }
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

/// Indices of the row groups whose `date` column statistics overlap `[start,
/// end]`. The long tables are `(date, symbol)`-sorted, so handing these to
/// [`ParquetRecordBatchReaderBuilder::with_row_groups`] **seeks past** the
/// out-of-window row groups without decoding them — the row-level `start` / `end`
/// checks still trim the (≤2) boundary groups. Row groups missing `date`
/// statistics are conservatively kept (must be scanned); with `start` and `end`
/// both `None` every group is kept.
fn date_row_groups_in_range(
    builder: &ParquetRecordBatchReaderBuilder<File>,
    date_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
) -> Vec<usize> {
    let meta = builder.metadata();
    let n = meta.num_row_groups();
    let Some(date_leaf) =
        builder.parquet_schema().columns().iter().position(|c| c.name() == date_column)
    else {
        return (0..n).collect();
    };
    (0..n)
        .filter(|&i| {
            // `date32` is physical INT32 (days); fall back to keeping the group if
            // statistics are absent or not the expected type.
            let Some(Statistics::Int32(v)) = meta.row_group(i).column(date_leaf).statistics()
            else {
                return true;
            };
            match (v.min_opt(), v.max_opt()) {
                (Some(&lo), Some(&hi)) => {
                    // `instant_from_days` is the exact per-row conversion, so this
                    // overlap test is consistent with the row-level filter.
                    let (lo_ts, hi_ts) = (instant_from_days(lo), instant_from_days(hi));
                    !(end.is_some_and(|e| lo_ts > e) || start.is_some_and(|s| hi_ts < s))
                }
                _ => true,
            }
        })
        .collect()
}

/// Count rows with `date` in `[start, end]` by scanning **only** the date column
/// of the row groups that overlap the window (out-of-window groups are skipped
/// via row-group statistics; the column is tiny once zstd-compressed, and the
/// table is `(date, symbol)`-sorted so the scan also stops at the first row past
/// `end`). This equals how many [`RowUpdate`]s [`read_panel`] emits in the window
/// — the progress-estimate unit.
pub(crate) fn count_rows_in_range(
    path: &str,
    date_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
) -> Result<usize, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let leaf = builder.schema().index_of(date_column).map_err(|e| e.to_string())?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), vec![leaf]);
    let row_groups = date_row_groups_in_range(&builder, date_column, start, end);
    let reader = builder
        .with_projection(mask)
        .with_row_groups(row_groups)
        .build()
        .map_err(|e| e.to_string())?;

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
            let ts = instant_from_days(dates.value(row));
            if end.is_some_and(|e| ts > e) {
                break 'outer;
            }
            if start.is_some_and(|s| ts < s) {
                continue;
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
