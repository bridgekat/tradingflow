//! Integration tests for [`tradingflow::sources::panel::parquet`] and
//! [`tradingflow::sources::panel::csv`]: a real file of each format, through
//! the full driver, observed per generation.
//!
//! The properties pinned here:
//!
//! * The pivot — each timestamp's rows land in the cells their index columns
//!   name; the signal marks exactly those *active* cells.
//! * Inactive cells (and null value cells) preserve the last emitted value —
//!   the cross-section is a carried state, and the signal is what says which
//!   cells are fresh.
//! * The timestamp column is a nanosecond timestamp, taken as [`Instant`]
//!   offsets as-is.
//! * The schema's order (not the file's) is the row order, and the schema
//!   must cover every label the table carries — there is no universe
//!   filtering in the source (a subset universe will later be a decode-level
//!   `RowFilter`).
//! * The half-open time range clips without carry-in.
//!
//! The two formats are the same source with a different decoder in front, so
//! the CSV tests pin what is CSV's own: the typing of a schemaless file, and
//! the parity of everything above with the Parquet reading of the same table.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, DictionaryArray, Float64Array, Int32Array, RecordBatch, TimestampNanosecondArray,
};
use arrow::datatypes::Int32Type;
use parquet::arrow::ArrowWriter;
use tradingflow::data::utils::{Axis, Schema};
use tradingflow::data::{ArrayView, Duration, Instant, SeriesView};
use tradingflow::graph::{Builder, Pool, Operator};
use tradingflow::operators::{series::record_all, signal};
use tradingflow::ports::{ArrayPort, SignalPort};
use tradingflow::sources::panel::{csv, parquet};
use tradingflow::time::UnixTime;

fn day(d: i32) -> Instant {
    Instant::from_offset(Duration::from_days(d as i64))
}

/// The `timestamp[ns]` column for the given day numbers — written through the
/// same [`day`] helper the assertions use, so the two stay consistent.
fn timestamps(days: impl IntoIterator<Item = i32>) -> TimestampNanosecondArray {
    TimestampNanosecondArray::from(
        days.into_iter()
            .map(|d| day(d).as_offset().as_nanos())
            .collect::<Vec<_>>(),
    )
}

/// Writes a `(date, symbol, close, volume)` long table; `symbol` is
/// dictionary-encoded, exactly as a low-cardinality string column comes out
/// of the crawler's Parquet files.
fn write_market_panel(path: &std::path::Path, rows: &[(i32, &str, Option<f64>, Option<f64>)]) {
    let dates = timestamps(rows.iter().map(|r| r.0));
    let symbols: DictionaryArray<Int32Type> = rows.iter().map(|r| r.1).collect();
    let closes = Float64Array::from(rows.iter().map(|r| r.2).collect::<Vec<_>>());
    let volumes = Float64Array::from(rows.iter().map(|r| r.3).collect::<Vec<_>>());
    let batch = RecordBatch::try_from_iter(vec![
        ("date", Arc::new(dates) as ArrayRef),
        ("symbol", Arc::new(symbols) as ArrayRef),
        ("close", Arc::new(closes) as ArrayRef),
        ("volume", Arc::new(volumes) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Writes the same `(date, symbol, close, volume)` long table as CSV. Dates
/// are written as epoch nanoseconds and a missing value as an empty field,
/// which is how a null arrives in a file that has no null of its own.
fn write_market_panel_csv(path: &std::path::Path, rows: &[(i32, &str, Option<f64>, Option<f64>)]) {
    let cell = |v: Option<f64>| v.map_or(String::new(), |v| v.to_string());
    let mut out = String::from("date,symbol,close,volume\n");
    for &(d, symbol, close, volume) in rows {
        let ns = day(d).as_offset().as_nanos();
        out.push_str(&format!("{ns},{symbol},{},{}\n", cell(close), cell(volume)));
    }
    std::fs::write(path, out).unwrap();
}

/// Snapshots a signal array as `f64` (1.0 / 0.0) so the per-generation mask
/// can be recorded next to the values it gates.
struct SignalFace<const N: usize>;

impl<const N: usize> Operator for SignalFace<N> {
    type Inputs = SignalPort<N>;
    type Outputs = ArrayPort<f64, N>;
    type Context = Instant;
    type State = tradingflow::data::Array<f64, N>;

    fn init(self, signal: ArrayView<'_, bool, N>) -> Self::State {
        tradingflow::data::Array::zeros(signal.extents())
    }

    fn reset<'a, 'b: 'a>(
        _: ArrayView<'a, bool, N>,
        state: &'b mut Self::State,
    ) -> ArrayView<'a, f64, N> {
        state.view()
    }

    fn compute<'a, 'b: 'a>(
        signal: ArrayView<'a, bool, N>,
        state: &'b mut Self::State,
        _: &Instant,
    ) -> ArrayView<'a, f64, N> {
        for (dst, &set) in state
            .data_mut()
            .iter_mut()
            .zip(signal.to_contiguous().iter())
        {
            *dst = f64::from(u8::from(set));
        }
        state.view()
    }
}

/// The rows of a recorded series, one `Vec` per generation.
fn rows<const N: usize>(v: SeriesView<'_, f64, N>) -> Vec<Vec<f64>> {
    let data = v.to_contiguous();
    let width = v.extents().iter().product::<usize>().max(1);
    data.chunks(width).map(<[f64]>::to_vec).collect()
}

fn assert_rows_eq(actual: &[Vec<f64>], expected: &[Vec<f64>], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: generation count");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.len(), e.len(), "{what}: row {i} width");
        for (j, (x, y)) in a.iter().zip(e).enumerate() {
            assert!(
                (x.is_nan() && y.is_nan()) || x == y,
                "{what}: row {i} element {j}: {x} != {y}"
            );
        }
    }
}

// ===========================================================================
// Parquet
// ===========================================================================

/// The core pivot, against a dictionary-encoded symbol column: per-date
/// cross-sections in schema order, both fields sharing the one signal.
/// Inactive cells — and null value cells in active rows — preserve the last
/// emitted value; only the signal says which cells are fresh.
#[tokio::test]
async fn panel_pivots_long_rows_into_per_date_cross_sections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.parquet");
    // Schema order is (AAA, BBB, CCC); the file carries BBB before AAA on
    // day 1 to pin that the schema, not the file, orders the rows. Day 2 has
    // no BBB row, and CCC's close is null while its volume is not.
    write_market_panel(
        &path,
        &[
            (1, "BBB", Some(20.0), Some(200.0)),
            (1, "AAA", Some(10.0), Some(100.0)),
            (2, "AAA", Some(11.0), None),
            (2, "CCC", None, Some(300.0)),
        ],
    );

    let schema = Schema::new(["AAA", "BBB", "CCC"]);
    let source = parquet(
        path.to_str().unwrap(),
        "date",
        [("symbol".into(), Axis::Labeled(schema))],
        vec!["close".into(), "volume".into()],
    );

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let mask_face = sc.op(SignalFace, sig);
    let mask_rec = sc.op(record_all(), (pulse, mask_face));
    let close_rec = sc.op(record_all(), (pulse, fields[0]));
    let volume_rec = sc.op(record_all(), (pulse, fields[1]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    let nan = f64::NAN;
    let masks = rows(g.view(mask_rec));
    let closes = rows(g.view(close_rec));
    let volumes = rows(g.view(volume_rec));
    assert_eq!(
        g.view(mask_rec).instants(),
        &[day(1), day(2)],
        "one generation per date, stamped with the date"
    );
    assert_rows_eq(&masks, &[vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]], "mask");
    assert_rows_eq(
        &closes,
        // Day 2: AAA fresh; BBB inactive, so it carries day 1's 20.0; CCC is
        // active but its close is null, so the cell keeps its initial NaN.
        &[vec![10.0, 20.0, nan], vec![11.0, 20.0, nan]],
        "close",
    );
    assert_rows_eq(
        &volumes,
        // Day 2: AAA is active with a *null* volume — the cell carries day
        // 1's 100.0, and only the signal (not the value) says it is stale-
        // free; BBB carries; CCC is fresh.
        &[vec![100.0, 200.0, nan], vec![100.0, 200.0, 300.0]],
        "volume",
    );
}

/// The window's end also ends the scan: dates past it exist in the file but
/// not for the graph — no empty generations, only absent ones.
#[tokio::test]
async fn an_out_of_window_date_produces_no_generation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.parquet");
    write_market_panel(
        &path,
        &[
            (1, "AAA", Some(10.0), Some(100.0)),
            (2, "AAA", Some(11.0), Some(110.0)),
            (3, "AAA", Some(12.0), Some(120.0)),
        ],
    );

    let source = parquet(
        path.to_str().unwrap(),
        "date",
        [("symbol".into(), Axis::Labeled(Schema::new(["AAA"])))],
        vec!["close".into()],
    )
    .with_time_range(None, Some(day(3)));

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(
        g.view(close_rec).instants(),
        &[day(1), day(2)],
        "the run ends at the window's end, with no empty day-3 generation"
    );
    assert_rows_eq(&rows(g.view(close_rec)), &[vec![10.0], vec![11.0]], "close");
}

/// The inclusive-exclusive time range clips both ends, with no carry-in of the
/// last pre-window value.
#[tokio::test]
async fn time_range_clips_without_carry_in() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.parquet");
    write_market_panel(
        &path,
        &[
            (1, "AAA", Some(10.0), Some(100.0)),
            (2, "AAA", Some(11.0), Some(110.0)),
            (3, "AAA", Some(12.0), Some(120.0)),
            (4, "AAA", Some(13.0), Some(130.0)),
        ],
    );

    let source = parquet(
        path.to_str().unwrap(),
        "date",
        [("symbol".into(), Axis::Labeled(Schema::new(["AAA"])))],
        vec!["close".into()],
    )
    .with_time_range(Some(day(2)), Some(day(4)));

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(g.view(close_rec).instants(), &[day(2), day(3)]);
    assert_rows_eq(&rows(g.view(close_rec)), &[vec![11.0], vec![12.0]], "close");
}

/// Timestamp-type dispatch: a `date32` column — the crawler's existing
/// format — converts on read, each day becoming its midnight [`Instant`],
/// identical to what a pre-converted `timestamp[ns]` column yields.
#[tokio::test]
async fn a_date32_timestamp_column_converts_on_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.parquet");
    let dates = arrow::array::Date32Array::from(vec![1, 2]);
    let closes = Float64Array::from(vec![10.0, 11.0]);
    let batch = RecordBatch::try_from_iter(vec![
        ("date", Arc::new(dates) as ArrayRef),
        ("close", Arc::new(closes) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = parquet(path.to_str().unwrap(), "date", [], vec!["close".into()]);

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(
        g.view(close_rec).instants(),
        &[day(1), day(2)],
        "day numbers arrive as their midnight instants"
    );
    assert_rows_eq(&rows(g.view(close_rec)), &[vec![10.0], vec![11.0]], "close");
}

/// Rank 0: no index columns at all — a single instrument's file, each value
/// column a scalar stream on the date cadence.
#[tokio::test]
async fn a_rank_zero_panel_is_a_scalar_stream() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ohlc.parquet");
    let dates = timestamps([1, 2, 3]);
    let closes = Float64Array::from(vec![Some(10.0), None, Some(12.0)]);
    let batch = RecordBatch::try_from_iter(vec![
        ("date", Arc::new(dates) as ArrayRef),
        ("close", Arc::new(closes) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = parquet(path.to_str().unwrap(), "date", [], vec!["close".into()]);

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(
        g.view(close_rec).instants(),
        &[day(1), day(2), day(3)],
        "every date pulses — a null close is a present row with missing data"
    );
    assert_rows_eq(
        &rows(g.view(close_rec)),
        // Day 2's close is null, so the cell carries day 1's value.
        &[vec![10.0], vec![10.0], vec![12.0]],
        "close",
    );
}

/// Rank 2: a named axis times a numeric axis. Each date's rows scatter into
/// a `[symbols, buckets]` cross-section.
#[tokio::test]
async fn a_rank_two_panel_scatters_on_both_axes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel2.parquet");
    let dates = timestamps([1, 1, 1, 2]);
    let symbols: DictionaryArray<Int32Type> =
        vec!["AAA", "AAA", "BBB", "BBB"].into_iter().collect();
    let buckets = Int32Array::from(vec![0, 1, 1, 0]);
    let values = Float64Array::from(vec![1.0, 2.0, 5.0, 7.0]);
    let batch = RecordBatch::try_from_iter(vec![
        ("date", Arc::new(dates) as ArrayRef),
        ("symbol", Arc::new(symbols) as ArrayRef),
        ("bucket", Arc::new(buckets) as ArrayRef),
        ("value", Arc::new(values) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let source = parquet(
        path.to_str().unwrap(),
        "date",
        [
            ("symbol".into(), Axis::Labeled(Schema::new(["AAA", "BBB"]))),
            ("bucket".into(), Axis::Fixed(2)),
        ],
        vec!["value".into()],
    );

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let value_rec = sc.op(record_all(), (pulse, fields[0]));
    let mask_face = sc.op(SignalFace, sig);
    let mask_rec = sc.op(record_all(), (pulse, mask_face));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    let nan = f64::NAN;
    assert_eq!(g.view(value_rec).instants(), &[day(1), day(2)]);
    // Row-major [2, 2]: [AAA/0, AAA/1, BBB/0, BBB/1]. On day 2 only BBB/0 is
    // fresh; the other cells carry day 1 (BBB/1 its 5.0, AAA/0 and AAA/1
    // theirs), and the never-written cell stays at its initial NaN on day 1.
    assert_rows_eq(
        &rows(g.view(value_rec)),
        &[vec![1.0, 2.0, nan, 5.0], vec![1.0, 2.0, 7.0, 5.0]],
        "value",
    );
    assert_rows_eq(
        &rows(g.view(mask_rec)),
        &[vec![1.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 0.0]],
        "mask",
    );
}

// ===========================================================================
// CSV
// ===========================================================================

/// The same table, the same graph, the same generations — only the decoder
/// differs. Empty fields stand in for the Parquet nulls, and the row count
/// the driver reports comes from the file's lines rather than a footer.
#[tokio::test]
async fn a_csv_panel_pivots_exactly_as_the_parquet_one_does() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.csv");
    write_market_panel_csv(
        &path,
        &[
            (1, "BBB", Some(20.0), Some(200.0)),
            (1, "AAA", Some(10.0), Some(100.0)),
            (2, "AAA", Some(11.0), None),
            (2, "CCC", None, Some(300.0)),
        ],
    );

    let schema = Schema::new(["AAA", "BBB", "CCC"]);
    let source = csv(
        path.to_str().unwrap(),
        "date",
        [("symbol".into(), Axis::Labeled(schema))],
        vec!["close".into(), "volume".into()],
    );

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let mask_face = sc.op(SignalFace, sig);
    let mask_rec = sc.op(record_all(), (pulse, mask_face));
    let close_rec = sc.op(record_all(), (pulse, fields[0]));
    let volume_rec = sc.op(record_all(), (pulse, fields[1]));

    let mut g = sc.build();
    assert_eq!(
        g.size_hint(),
        Some(4),
        "the four data rows, less the header"
    );
    g.run(&mut Pool::new(0), |_, _| {}).await;

    let nan = f64::NAN;
    assert_eq!(g.view(mask_rec).instants(), &[day(1), day(2)]);
    assert_rows_eq(
        &rows(g.view(mask_rec)),
        &[vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]],
        "mask",
    );
    assert_rows_eq(
        &rows(g.view(close_rec)),
        &[vec![10.0, 20.0, nan], vec![11.0, 20.0, nan]],
        "close",
    );
    assert_rows_eq(
        &rows(g.view(volume_rec)),
        &[vec![100.0, 200.0, nan], vec![100.0, 200.0, 300.0]],
        "volume",
    );
}

/// The date column is the one column a CSV source types from the file, and
/// the three spellings of a date all name the same instants.
#[tokio::test]
async fn a_csv_date_column_is_typed_by_how_it_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let epoch = |d: i32| day(d).as_offset().as_nanos();
    let cases = [
        (
            "calendar dates",
            "1970-01-02".to_string(),
            "1970-01-03".to_string(),
        ),
        (
            "timestamps",
            "1970-01-02 00:00:00".to_string(),
            "1970-01-03 00:00:00".to_string(),
        ),
        (
            "epoch nanoseconds",
            epoch(1).to_string(),
            epoch(2).to_string(),
        ),
    ];

    for (what, first, second) in cases {
        let path = dir.path().join(format!("{}.csv", what.replace(' ', "_")));
        std::fs::write(&path, format!("date,close\n{first},10.0\n{second},11.0\n")).unwrap();

        let source = csv(path.to_str().unwrap(), "date", [], vec!["close".into()]);
        let mut sc = Builder::new(UnixTime);
        let (sig, fields) = sc.source(source);
        let pulse = sc.op(signal::any(), sig);
        let close_rec = sc.op(record_all(), (pulse, fields[0]));

        let mut g = sc.build();
        g.run(&mut Pool::new(0), |_, _| {}).await;

        assert_eq!(g.view(close_rec).instants(), &[day(1), day(2)], "{what}");
        assert_rows_eq(&rows(g.view(close_rec)), &[vec![10.0], vec![11.0]], what);
    }
}

/// An index column is typed by its axis, not by its contents: numeric symbols
/// are labels, and the schema — not their numeric value — places them.
#[tokio::test]
async fn a_numeric_label_column_is_read_as_labels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.csv");
    let epoch = |d: i32| day(d).as_offset().as_nanos();
    std::fs::write(
        &path,
        format!(
            "date,symbol,close\n\
             {d1},600001,20.0\n\
             {d1},600000,10.0\n\
             {d2},600000,11.0\n",
            d1 = epoch(1),
            d2 = epoch(2),
        ),
    )
    .unwrap();

    let source = csv(
        path.to_str().unwrap(),
        "date",
        [(
            "symbol".into(),
            Axis::Labeled(Schema::new(["600000", "600001"])),
        )],
        vec!["close".into()],
    );

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(g.view(close_rec).instants(), &[day(1), day(2)]);
    // Schema order, not file order and not numeric order: 600000 first.
    assert_rows_eq(
        &rows(g.view(close_rec)),
        &[vec![10.0, 20.0], vec![11.0, 20.0]],
        "close",
    );
}

/// Rank 2 over CSV: a named axis times a numeric one, the numeric column read
/// as integers because its axis says so.
#[tokio::test]
async fn a_rank_two_csv_panel_scatters_on_both_axes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel2.csv");
    let epoch = |d: i32| day(d).as_offset().as_nanos();
    std::fs::write(
        &path,
        format!(
            "date,symbol,bucket,value\n\
             {d1},AAA,0,1.0\n\
             {d1},AAA,1,2.0\n\
             {d1},BBB,1,5.0\n\
             {d2},BBB,0,7.0\n",
            d1 = epoch(1),
            d2 = epoch(2),
        ),
    )
    .unwrap();

    let source = csv(
        path.to_str().unwrap(),
        "date",
        [
            ("symbol".into(), Axis::Labeled(Schema::new(["AAA", "BBB"]))),
            ("bucket".into(), Axis::Fixed(2)),
        ],
        vec!["value".into()],
    );

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let value_rec = sc.op(record_all(), (pulse, fields[0]));
    let mask_face = sc.op(SignalFace, sig);
    let mask_rec = sc.op(record_all(), (pulse, mask_face));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    let nan = f64::NAN;
    assert_eq!(g.view(value_rec).instants(), &[day(1), day(2)]);
    assert_rows_eq(
        &rows(g.view(value_rec)),
        &[vec![1.0, 2.0, nan, 5.0], vec![1.0, 2.0, 7.0, 5.0]],
        "value",
    );
    assert_rows_eq(
        &rows(g.view(mask_rec)),
        &[vec![1.0, 1.0, 0.0, 1.0], vec![0.0, 0.0, 1.0, 0.0]],
        "mask",
    );
}

/// The window clips a CSV scan the same way, at both ends: no carry-in of the
/// last pre-window value, and no generation past the end.
#[tokio::test]
async fn a_csv_time_range_clips_without_carry_in() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panel.csv");
    write_market_panel_csv(
        &path,
        &[
            (1, "AAA", Some(10.0), Some(100.0)),
            (2, "AAA", Some(11.0), Some(110.0)),
            (3, "AAA", Some(12.0), Some(120.0)),
            (4, "AAA", Some(13.0), Some(130.0)),
        ],
    );

    let source = csv(
        path.to_str().unwrap(),
        "date",
        [("symbol".into(), Axis::Labeled(Schema::new(["AAA"])))],
        vec!["close".into()],
    )
    .with_time_range(Some(day(2)), Some(day(4)));

    let mut sc = Builder::new(UnixTime);
    let (sig, fields) = sc.source(source);
    let pulse = sc.op(signal::any(), sig);
    let close_rec = sc.op(record_all(), (pulse, fields[0]));

    let mut g = sc.build();
    g.run(&mut Pool::new(0), |_, _| {}).await;

    assert_eq!(g.view(close_rec).instants(), &[day(2), day(3)]);
    assert_rows_eq(&rows(g.view(close_rec)), &[vec![11.0], vec![12.0]], "close");
}
