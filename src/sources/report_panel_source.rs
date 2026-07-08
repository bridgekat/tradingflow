//! Cross-sectional panel source for **financial-report** long tables.
//!
//! Like [`ParquetPanelSource`](super::ParquetPanelSource) it pivots a long table
//! into one wide `[N, R]` cross-section per event date (StackSync semantics — each
//! cross-section reflects only that date's reports, `NaN` elsewhere; the
//! carry-forward is the downstream [`Stack`](crate::operators::Stack)'s job). It
//! additionally understands the report layout — two date columns (`date` =
//! period-end, `notice_date` = publication, nullable) — and the
//! [`FinancialReportSource`](super::stocks::FinancialReportSource) point-in-time
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
//! [`with_report_date`](ReportPanelSource::with_report_date) prepends
//! `[year, day_of_year]` of the **report** date (for
//! [`Annualize`](crate::operators::Annualize)), exactly as `FinancialReportSource`. A
//! per-stock pipeline is recovered downstream by `Select` + a NaN `Filter`.

use std::collections::HashMap;
use std::fs::File;

use tokio::sync::mpsc;

use arrow::array::{Array as _, Date32Array};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::parquet_panel_source::{
    PanelState, RowUpdate, count_rows_in_range, instant_from_days, nan_panel, panel_write,
    report_year_and_doy, resolve_symbols,
};
use crate::{Array, Duration, Instant, Source};

/// Historical-only panel source for financial-report long tables. See module docs.
#[derive(Clone)]
pub struct ReportPanelSource {
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

impl ReportPanelSource {
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

/// One parsed report row, keyed by its event timestamp.
struct ReportRow {
    key_ts: Instant,
    report_days: i32,
    ui: usize,
    values: Vec<f64>,
}

impl Source for ReportPanelSource {
    type Event = Vec<RowUpdate>;
    type Output = Array<f64, 2>;
    type State = PanelState;

    fn estimated_event_count(&self) -> Option<usize> {
        // Progress is in emitted long-table rows. The effective-date emits (after
        // the retrospective-drop) are bounded by the rows in `[start, end]` on the
        // report-date timeline — a close proxy (reports are a small minority of
        // total events). On any read error fall back to `Some(0)`.
        Some(
            count_rows_in_range(&self.path, &self.report_date_column, self.start, self.end).unwrap_or(0),
        )
    }

    fn initial(&self) -> Array<f64, 2> {
        nan_panel(self.out_shape())
    }

    fn init(&self) -> (mpsc::Receiver<(Instant, Vec<RowUpdate>)>, PanelState) {
        // One item per tick (a batch of that date's reports); small buffer.
        let (hist_tx, hist_rx) = mpsc::channel(16);
        let cfg = self.clone();

        tokio::task::spawn_blocking(move || {
            if let Err(e) = read_reports(&cfg, &hist_tx) {
                eprintln!("ReportPanelSource error ({}): {e}", cfg.path);
            }
        });

        (hist_rx, PanelState::default())
    }

    fn write(state: &mut PanelState, batch: Vec<RowUpdate>, output: &mut Array<f64, 2>, ts: Instant) -> usize {
        panel_write(state, batch, output, ts)
    }
}

fn read_reports(
    cfg: &ReportPanelSource,
    hist_tx: &mpsc::Sender<(Instant, Vec<RowUpdate>)>,
) -> Result<(), String> {
    let value_offset = if cfg.with_report_date { 2 } else { 0 };
    let r = value_offset + cfg.value_columns.len();
    let sym_index: HashMap<&str, usize> =
        cfg.symbols.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    let file = File::open(&cfg.path).map_err(|e| format!("open: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| e.to_string())?;
    let schema = builder.schema();
    let mut needed: Vec<&str> =
        vec![cfg.report_date_column.as_str(), cfg.notice_date_column.as_str(), cfg.symbol_column.as_str()];
    needed.extend(cfg.value_columns.iter().map(|s| s.as_str()));
    let leaf_indices: Vec<usize> = needed
        .iter()
        .map(|name| schema.index_of(name).map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let mask = ProjectionMask::leaves(builder.parquet_schema(), leaf_indices);
    let reader = builder.with_projection(mask).build().map_err(|e| e.to_string())?;

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
            .ok_or_else(|| format!("report date column {:?} is not date32", cfg.report_date_column))?;
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
            .map(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>().unwrap())
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
                .map(|va| if va.is_null(row) { f64::NAN } else { va.value(row) })
                .collect();
            rows.push(ReportRow { key_ts, report_days, ui, values });
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
            if let Some(t) = cur_ts {
                if hist_tx.blocking_send((t, std::mem::take(&mut tick))).is_err() {
                    return Ok(());
                }
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
        tick.push(RowUpdate { row: row.ui, vals: payload.into_boxed_slice() });
    }
    // Flush the final (also last in-window) tick.
    if let Some(t) = cur_ts {
        if !tick.is_empty() {
            let _ = hist_tx.blocking_send((t, tick));
        }
    }
    Ok(())
}
