//! Cross-sectional panel sources over long-format Parquet tables.
//!
//! Both read a **long** table and emit one **wide cross-section** per event
//! date — an `Array<f64>` of shape `[N, K]` over a fixed `symbols` universe,
//! rows for absent symbols `NaN`:
//!
//! * [`ParquetPanelSource`] — the general `(date, symbol, <values…>)` panel.
//! * [`ParquetFinancialReportPanelSource`] — the financial-report variant,
//!   which additionally understands the two-date report layout and
//!   point-in-time (effective-date) alignment.
//!
//! # The general panel
//!
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
//! downstream [`Stack`](crate::operators::structural::Stack) / [`StackSync`](crate::operators::structural::StackSync)
//! operators — not the source. With `with_time_range`, rows before `start` are
//! simply skipped (no last-value-before-`start` is carried in).
//!
//! # Irregular kinds & message-passing operators
//!
//! For daily prices every symbol ticks on (almost) every date, so the panel is
//! dense. For **irregular** kinds (dividends, financial reports) the panel emits
//! at the *union* of all symbols' event dates — which need not include every
//! trading day. A per-stock `Select` therefore still fires on that union cadence
//! with `NaN` where the stock had no row; a [`Filter`](crate::operators::structural::Filter) that
//! drops the all-`NaN` ("no data") rows recovers that stock's true event stream,
//! so message-passing operators (e.g. [`ForwardAdjust`](crate::operators::stocks::ForwardAdjust))
//! see each real event exactly once — reproducing the per-symbol stream.
//!
//! # Timestamps
//!
//! `date` is a Parquet `date32` (days since 1970-01-01). The event timestamp is
//! the day's midnight instant (`EPOCH + days · 24 h`), so a panel→select→filter
//! stream is timestamp-aligned across kinds.
//!
//! Requires a tokio runtime when added to a scenario (the Parquet scan runs on a
//! `spawn_blocking` task feeding the historical channel with back-pressure).
//!
//! # The financial-report panel
//!
//!
//! Like [`ParquetPanelSource`] it pivots a long table
//! into one wide `[N, R]` cross-section per event date (StackSync semantics — each
//! cross-section reflects only that date's reports, `NaN` elsewhere; the
//! carry-forward is the downstream [`Stack`](crate::operators::structural::Stack)'s job). It
//! additionally understands the report layout — two date columns (`date` =
//! period-end, `notice_date` = publication, nullable) — and point-in-time
//! semantics:
//!
//! * **`use_effective_date = false`** (default) — events fire on the report
//!   `date`. The table is already `(date, symbol)`-sorted, so this is a straight
//!   scan (same as a `ParquetPanelSource` on `date`).
//! * **`use_effective_date = true`** — events fire on the *effective date*
//!   `e = max(date, notice_date)` (with `notice_fallback` added to `date` when
//!   `notice_date` is null), so reports are not visible until published. Rows are
//!   **reordered by `e`**, and **retrospective updates are dropped** (per symbol,
//!   in `e` order, keep a report only if its report `date` advances that symbol's
//!   high-water mark). This is the look-ahead-safe alignment that backtests want.
//!
//! The financial tables are small, so this is a **load-and-sort** entirely inside
//! the source; the engine sees a clean timestamp-non-decreasing stream.
//!
//! [`with_report_date`](ParquetFinancialReportPanelSource::with_report_date) prepends
//! `[year, day_of_year]` of the **report** date (for
//! [`Annualize`](crate::operators::stocks::Annualize)). A per-stock pipeline is recovered
//! downstream by `Select` + a NaN `Filter`.

use std::collections::HashMap;
use std::fs::File;

use arrow::array::{Array as _, ArrayRef, Date32Array, DictionaryArray, Int32Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Int32Type};
use futures::stream::Stream;
use hifitime::{Duration as HfDuration, Epoch};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::statistics::Statistics;
use tokio::sync::mpsc;

use super::receiver_stream;
use crate::data::{Array, ArrayView, Duration, Instant};
use crate::graph::{Event, Source};
use crate::ports::ArrayPort;

// ===========================================================================
// ParquetPanelSource — the general (date, symbol, values...) panel.
// ===========================================================================

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
    pub fn with_columns(
        mut self,
        date_column: impl Into<String>,
        symbol_column: impl Into<String>,
    ) -> Self {
        self.date_column = date_column.into();
        self.symbol_column = symbol_column.into();
        self
    }

    /// Emitted element shape, `[N, value_columns]`.
    pub fn out_shape(&self) -> [usize; 2] {
        [self.symbols.len(), self.value_columns.len()]
    }
}

/// A cross-sectional panel over the long-format Parquet table at `path`,
/// emitting one `[symbols, value_columns]` cross-section per date. Chain
/// [`with_time_range`](ParquetPanelSource::with_time_range) /
/// [`with_columns`](ParquetPanelSource::with_columns) to bound the dates or
/// rename the key columns.
pub fn parquet_panel_source(
    path: impl Into<String>,
    value_columns: Vec<String>,
    symbols: Vec<String>,
) -> ParquetPanelSource {
    ParquetPanelSource::new(path, value_columns, symbols)
}

/// `date32` days since 1970-01-01 → the day's UTC-midnight [`Epoch`] (hifitime).
fn epoch_from_days(days: i32) -> Epoch {
    Epoch::from_unix_duration(HfDuration::from_truncated_nanoseconds(
        days as i64 * 86_400 * 1_000_000_000,
    ))
}

/// `date32` days → event [`Instant`] (the day's midnight, naive).
fn instant_from_days(days: i32) -> Instant {
    Instant::from_offset(Duration::from_days(days as i64))
}

/// `(year, day_of_year)` (1-based) for a report date via hifitime
/// (`Epoch::year_days_of_year() + 1`), so [`Annualize`](crate::operators::stocks::Annualize)
/// consumes it directly. Used by [`ParquetFinancialReportPanelSource`](super::ParquetFinancialReportPanelSource).
fn report_year_and_doy(days: i32) -> (f64, f64) {
    let (year, day_of_year) = epoch_from_days(days).year_days_of_year();
    (year as f64, day_of_year + 1.0)
}

/// One narrow-table row's update to a wide `[N, K]` cross-section cell: symbol
/// `row`'s `K` values (NaN for null cells). The panel sources **batch a whole
/// tick's rows** into one channel item (`Vec<RowUpdate>`) — one message per
/// distinct date rather than per row, amortising the channel / event-loop cost —
/// and `panel_write` applies them per event (still the per-row state logic) to
/// reassemble the cross-section, returning the batch size so the run's event
/// count stays equal to the long-table row count.
pub struct RowUpdate {
    pub row: usize,
    pub vals: Box<[f64]>,
}

/// The panel writer's captured state: the timestamp of the tick being
/// assembled and the rows it dirtied. When the timestamp **strictly
/// advances**, `panel_write` NaN-clears those rows first — reproducing the
/// per-tick "only this date's rows" cross-section (pure StackSync, no carry).
#[derive(Default)]
pub struct PanelState {
    last_ts: Option<Instant>,
    dirty: Vec<usize>,
}

/// Shared writer body for both panel sources: apply one tick's batch.
/// On a new tick (the batch's `ts` strictly past the last) clear the previous
/// tick's rows first, then write each row's `K` values **per event**. Every row
/// in a batch shares the tick's `ts` and the source's `K` (taken from the first
/// row, so it works for both 2-D `[N, K]` and squeezed `[N]` cells). Returns the
/// number of rows applied — the per-event count the run reports.
/// The all-NaN initial panel: the per-tick `write` only sets rows that have
/// an event, so an unwritten row must read as `NaN` (not `0.0`) for the
/// per-stock `Filter` to drop it.
fn nan_panel(shape: [usize; 2]) -> Array<f64, 2> {
    Array::from_parts(shape, vec![f64::NAN; shape.iter().product()].into())
}

fn panel_write(
    state: &mut PanelState,
    ts: Instant,
    batch: Vec<RowUpdate>,
    output: &mut Array<f64, 2>,
) -> usize {
    let Some(first) = batch.first() else { return 0 };
    let k = first.vals.len();
    let buf = output.data_mut();
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

impl Source for ParquetPanelSource {
    type Instant = Instant;
    type Payload = Vec<RowUpdate>;
    type Outputs = ArrayPort<f64, 2>;
    type State = (PanelState, Array<f64, 2>);

    fn size_hint(&self) -> Option<usize> {
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

    fn init(
        self,
    ) -> (
        Self::State,
        impl Stream<Item = Event<Vec<RowUpdate>, Instant>> + 'static,
    ) {
        // Each item is now a whole tick's rows, so a small buffer pipelines plenty
        // of ticks ahead while bounding the in-flight row memory.
        let state = (PanelState::default(), nan_panel(self.out_shape()));
        let (hist_tx, hist_rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            if let Err(e) = read_panel(&self, &hist_tx) {
                eprintln!("ParquetPanelSource error ({}): {e}", self.path);
            }
        });
        (state, receiver_stream(hist_rx))
    }

    fn output(state: &mut Self::State) -> (bool, ArrayView<'_, f64, 2>) {
        (true, state.1.view())
    }

    fn write(payload: Vec<RowUpdate>, instant: Instant, state: &mut Self::State) -> usize {
        let (state, output) = state;
        panel_write(state, instant, payload, output)
    }
}

/// Sequential scan: read row groups in `(date, symbol)` order and emit **one
/// batch per distinct date** — a `Vec<RowUpdate>` of that date's in-universe,
/// in-window rows, timestamped by the date. Buffering rows until the date
/// changes (and flushing the last date after the scan) keeps one channel message
/// per tick; `panel_write` then reassembles each date's `[N, K]` cross-section
/// and clears the previous date's rows when the timestamp advances.
#[expect(
    clippy::needless_range_loop,
    reason = "row drives the arrow column accessors (`dates.value(row)`), not just a slice"
)]
fn read_panel(
    cfg: &ParquetPanelSource,
    hist_tx: &mpsc::Sender<(Vec<RowUpdate>, Instant)>,
) -> Result<(), String> {
    let k = cfg.value_columns.len();
    let sym_index: HashMap<&str, usize> = cfg
        .symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

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
            .map(|a| {
                a.as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap()
            })
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
                if let Some(t) = cur_ts
                    && hist_tx
                        .blocking_send((std::mem::take(&mut tick), t))
                        .is_err()
                {
                    return Ok(());
                }
                cur_ts = Some(ts);
            }
            let mut payload = vec![f64::NAN; k];
            for (vi, va) in vals.iter().enumerate() {
                if !va.is_null(row) {
                    payload[vi] = va.value(row);
                }
            }
            tick.push(RowUpdate {
                row: ui,
                vals: payload.into_boxed_slice(),
            });
        }
    }
    // Flush the final tick (also the last in-window tick when we broke on `end`).
    if let Some(t) = cur_ts
        && !tick.is_empty()
    {
        let _ = hist_tx.blocking_send((tick, t));
    }
    Ok(())
}

/// Total row count from the parquet footer metadata — O(1), no row decode.
/// The number of rows a full scan (no `end` bound) reads.
fn parquet_num_rows(path: &str) -> Result<usize, String> {
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
    let Some(date_leaf) = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|c| c.name() == date_column)
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
fn count_rows_in_range(
    path: &str,
    date_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
) -> Result<usize, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let leaf = builder
        .schema()
        .index_of(date_column)
        .map_err(|e| e.to_string())?;
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
fn resolve_symbols(
    sym_col: &ArrayRef,
    sym_index: &HashMap<&str, usize>,
) -> Result<Vec<Option<usize>>, String> {
    if let Some(dict) = sym_col
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
    {
        let values = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("dictionary values are not utf8")?;
        // Dictionary entry -> universe index (computed once; dictionary is small).
        let entry_uni: Vec<Option<usize>> = (0..values.len())
            .map(|key| sym_index.get(values.value(key)).copied())
            .collect();
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
            .map(|row| {
                if sa.is_null(row) {
                    None
                } else {
                    sym_index.get(sa.value(row)).copied()
                }
            })
            .collect())
    }
}

// ===========================================================================
// ParquetFinancialReportPanelSource — the report panel (two dates, PIT).
// ===========================================================================

/// Historical-only panel source for financial-report long tables. See module docs.
#[derive(Clone)]
pub struct ParquetFinancialReportPanelSource {
    path: String,
    value_columns: Vec<String>,
    symbols: Vec<String>,
    with_report_date: bool,
    use_effective_date: bool,
    notice_fallback: Duration,
    report_date_column: String,
    notice_date_column: String,
    symbol_column: String,
    start: Option<Instant>,
    end: Option<Instant>,
}

impl ParquetFinancialReportPanelSource {
    /// Create a report panel source. `value_columns` are emitted in order (each
    /// cast to `f64`); `symbols` is the universe row order. Defaults to report-date
    /// alignment (`use_effective_date = false`).
    pub fn new(path: impl Into<String>, value_columns: Vec<String>, symbols: Vec<String>) -> Self {
        Self {
            path: path.into(),
            value_columns,
            symbols,
            with_report_date: false,
            use_effective_date: false,
            notice_fallback: Duration::ZERO,
            report_date_column: "date".into(),
            notice_date_column: "notice_date".into(),
            symbol_column: "symbol".into(),
            start: None,
            end: None,
        }
    }

    /// Prepend `[year, day_of_year]` of the report date to each row (for
    /// `Annualize`); emitted shape becomes `[N, 2 + value_columns]`.
    pub fn with_report_date(mut self, with_report_date: bool) -> Self {
        self.with_report_date = with_report_date;
        self
    }

    /// Fire events on the **effective date** `max(date, notice_date)` (with
    /// `notice_fallback` added to `date` when `notice_date` is null), reordering
    /// and dropping retrospective updates. Without this, events fire on the
    /// report `date`. This is the look-ahead-safe alignment for backtesting.
    pub fn use_effective_date(mut self, notice_fallback: Duration) -> Self {
        self.use_effective_date = true;
        self.notice_fallback = notice_fallback;
        self
    }

    /// Restrict emitted cross-sections to event timestamps in `[start, end]`.
    pub fn with_time_range(mut self, start: Option<Instant>, end: Option<Instant>) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Override the report-date / notice-date / symbol column names.
    pub fn with_columns(
        mut self,
        report_date_column: impl Into<String>,
        notice_date_column: impl Into<String>,
        symbol_column: impl Into<String>,
    ) -> Self {
        self.report_date_column = report_date_column.into();
        self.notice_date_column = notice_date_column.into();
        self.symbol_column = symbol_column.into();
        self
    }

    /// Columns per cross-section row (`2 + value_columns` under `with_report_date`).
    pub fn row_width(&self) -> usize {
        (if self.with_report_date { 2 } else { 0 }) + self.value_columns.len()
    }

    /// Emitted element shape, `[N, row_width]`.
    pub fn out_shape(&self) -> [usize; 2] {
        [self.symbols.len(), self.row_width()]
    }
}

/// A financial-report panel over the long-format Parquet table at `path`,
/// report-date aligned. Chain
/// [`use_effective_date`](ParquetFinancialReportPanelSource::use_effective_date)
/// for point-in-time alignment,
/// [`with_report_date`](ParquetFinancialReportPanelSource::with_report_date) to
/// prepend `[year, day_of_year]` (for
/// [`annualize`](crate::operators::stocks::annualize)), or
/// [`with_time_range`](ParquetFinancialReportPanelSource::with_time_range) /
/// [`with_columns`](ParquetFinancialReportPanelSource::with_columns).
pub fn parquet_financial_report_panel_source(
    path: impl Into<String>,
    value_columns: Vec<String>,
    symbols: Vec<String>,
) -> ParquetFinancialReportPanelSource {
    ParquetFinancialReportPanelSource::new(path, value_columns, symbols)
}

/// One parsed report row, keyed by its event timestamp.
struct ReportRow {
    key_ts: Instant,
    report_days: i32,
    ui: usize,
    values: Vec<f64>,
}

impl Source for ParquetFinancialReportPanelSource {
    type Instant = Instant;
    type Payload = Vec<RowUpdate>;
    type Outputs = ArrayPort<f64, 2>;
    type State = (PanelState, Array<f64, 2>);

    fn size_hint(&self) -> Option<usize> {
        // Progress is in emitted long-table rows. The effective-date emits (after
        // the retrospective-drop) are bounded by the rows in `[start, end]` on the
        // report-date timeline — a close proxy (reports are a small minority of
        // total events). On any read error fall back to `Some(0)`.
        Some(
            count_rows_in_range(&self.path, &self.report_date_column, self.start, self.end)
                .unwrap_or(0),
        )
    }

    fn init(
        self,
    ) -> (
        Self::State,
        impl Stream<Item = Event<Vec<RowUpdate>, Instant>> + 'static,
    ) {
        // One item per tick (a batch of that date's reports); small buffer.
        let state = (PanelState::default(), nan_panel(self.out_shape()));
        let (hist_tx, hist_rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || {
            if let Err(e) = read_reports(&self, &hist_tx) {
                eprintln!(
                    "ParquetFinancialReportPanelSource error ({}): {e}",
                    self.path
                );
            }
        });
        (state, receiver_stream(hist_rx))
    }

    fn output(state: &mut Self::State) -> (bool, ArrayView<'_, f64, 2>) {
        (true, state.1.view())
    }

    fn write(payload: Vec<RowUpdate>, instant: Instant, state: &mut Self::State) -> usize {
        let (state, output) = state;
        panel_write(state, instant, payload, output)
    }
}

#[expect(
    clippy::needless_range_loop,
    reason = "row drives the arrow column accessors (`dates.value(row)`), not just a slice"
)]
fn read_reports(
    cfg: &ParquetFinancialReportPanelSource,
    hist_tx: &mpsc::Sender<(Vec<RowUpdate>, Instant)>,
) -> Result<(), String> {
    let value_offset = if cfg.with_report_date { 2 } else { 0 };
    let r = value_offset + cfg.value_columns.len();
    let sym_index: HashMap<&str, usize> = cfg
        .symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let file = File::open(&cfg.path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let schema = builder.schema();
    let mut needed: Vec<&str> = vec![
        cfg.report_date_column.as_str(),
        cfg.notice_date_column.as_str(),
        cfg.symbol_column.as_str(),
    ];
    needed.extend(cfg.value_columns.iter().map(|s| s.as_str()));
    let leaf_indices: Vec<usize> = needed
        .iter()
        .map(|name| schema.index_of(name).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), leaf_indices);
    let reader = builder
        .with_projection(mask)
        .build()
        .map_err(|e| e.to_string())?;

    // 1. Read every row, keyed by its event timestamp.
    let mut rows: Vec<ReportRow> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|e| e.to_string())?;

        let report_col = batch
            .column_by_name(&cfg.report_date_column)
            .ok_or_else(|| format!("missing report date column {:?}", cfg.report_date_column))?;
        let report_dates = report_col
            .as_any()
            .downcast_ref::<Date32Array>()
            .ok_or_else(|| {
                format!(
                    "report date column {:?} is not date32",
                    cfg.report_date_column
                )
            })?;
        let notice_dates = batch
            .column_by_name(&cfg.notice_date_column)
            .and_then(|c| c.as_any().downcast_ref::<Date32Array>());

        let sym_col = batch
            .column_by_name(&cfg.symbol_column)
            .ok_or_else(|| format!("missing symbol column {:?}", cfg.symbol_column))?;
        let row_uni = resolve_symbols(sym_col, &sym_index)?;

        let val_refs: Vec<_> = cfg
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
            .map(|a| {
                a.as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap()
            })
            .collect();

        for row in 0..batch.num_rows() {
            let Some(ui) = row_uni[row] else { continue };
            let report_days = report_dates.value(row);
            let report_ts = instant_from_days(report_days);
            let key_ts = if cfg.use_effective_date {
                let notice_ts = match notice_dates {
                    Some(nd) if !nd.is_null(row) => instant_from_days(nd.value(row)),
                    _ => report_ts + cfg.notice_fallback,
                };
                report_ts.max(notice_ts)
            } else {
                report_ts
            };
            let values: Vec<f64> = vals
                .iter()
                .map(|va| {
                    if va.is_null(row) {
                        f64::NAN
                    } else {
                        va.value(row)
                    }
                })
                .collect();
            rows.push(ReportRow {
                key_ts,
                report_days,
                ui,
                values,
            });
        }
    }

    // 2. Effective-date mode: drop retrospective updates per symbol (walking in
    //    `e` order, keep only reports whose report date advances the high-water
    //    mark).
    if cfg.use_effective_date {
        let mut by_symbol: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, row) in rows.iter().enumerate() {
            by_symbol.entry(row.ui).or_default().push(i);
        }
        let mut keep = vec![false; rows.len()];
        for idxs in by_symbol.values_mut() {
            idxs.sort_by_key(|&i| rows[i].key_ts);
            let mut hwm = i32::MIN;
            for &i in idxs.iter() {
                if rows[i].report_days > hwm {
                    hwm = rows[i].report_days;
                    keep[i] = true;
                }
            }
        }
        let mut k = 0;
        rows.retain(|_| {
            let keep_it = keep[k];
            k += 1;
            keep_it
        });
    }

    // 3. Emit one batch per distinct effective date (`key_ts`), accumulating that
    //    date's kept reports into a `Vec<RowUpdate>`. The downstream `panel_write`
    //    reassembles each tick's cross-section (StackSync — only that tick's
    //    reports; absent symbols NaN) and clears the previous tick when the
    //    timestamp advances. Rows before `start` are skipped.
    rows.sort_by_key(|row| row.key_ts);
    let (start, end) = (cfg.start, cfg.end);
    let mut cur_ts: Option<Instant> = None;
    let mut tick: Vec<RowUpdate> = Vec::new();
    for row in &rows {
        let ts = row.key_ts;
        if end.is_some_and(|e| ts > e) {
            break;
        }
        if start.is_some_and(|s| ts < s) {
            continue;
        }
        if cur_ts != Some(ts) {
            if let Some(t) = cur_ts
                && hist_tx
                    .blocking_send((std::mem::take(&mut tick), t))
                    .is_err()
            {
                return Ok(());
            }
            cur_ts = Some(ts);
        }
        let mut payload = vec![f64::NAN; r];
        if cfg.with_report_date {
            let (year, doy) = report_year_and_doy(row.report_days);
            payload[0] = year;
            payload[1] = doy;
        }
        for (vi, v) in row.values.iter().enumerate() {
            payload[value_offset + vi] = *v;
        }
        tick.push(RowUpdate {
            row: row.ui,
            vals: payload.into_boxed_slice(),
        });
    }
    // Flush the final (also last in-window) tick.
    if let Some(t) = cur_ts
        && !tick.is_empty()
    {
        let _ = hist_tx.blocking_send((tick, t));
    }
    Ok(())
}
