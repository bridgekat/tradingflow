use arrow::array::{Array as _, ArrayRef, TimestampNanosecondArray};
use arrow::compute::{CastOptions, cast, cast_with_options};
use arrow::datatypes::{DataType, TimeUnit};
use bumpalo::Bump;
use futures::stream::Stream;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::data::utils::Axis;
use crate::data::utils::arrow::{read_index_columns, read_value_column};
use crate::data::{Array, ArrayView, Duration, Instant};
use crate::graph::{Event, Source};
use crate::ports::{ArrayPorts, SignalPort};

/// Source signature for [`parquet()`].
pub struct Parquet<const N: usize> {
    path: String,
    date_column: String,
    index_columns: [(String, Axis); N],
    value_columns: Vec<String>,
    start: Option<Instant>,
    end: Option<Instant>,
}

impl<const N: usize> Parquet<N> {
    pub fn new(
        path: impl Into<String>,
        index_columns: [(String, Axis); N],
        value_columns: Vec<String>,
    ) -> Self {
        Self {
            path: path.into(),
            date_column: "date".into(),
            index_columns,
            value_columns,
            start: None,
            end: None,
        }
    }

    /// Overrides the date column name (default `"date"`).
    pub fn with_date_column(mut self, date_column: impl Into<String>) -> Self {
        self.date_column = date_column.into();
        self
    }

    /// Restricts emission to dates in `[start, end)`. Rows before `start` are
    /// skipped instead of carried in.
    pub fn with_time_range(mut self, start: Option<Instant>, end: Option<Instant>) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    fn extents(&self) -> [usize; N] {
        std::array::from_fn(|j| {
            let (_, axis) = &self.index_columns[j];
            match axis {
                Axis::Labeled(schema) => schema.len(),
                Axis::Fixed(extent) => *extent,
                Axis::None => panic!("axis {j} has no known extent"),
            }
        })
    }
}

/// Runtime state for [`Parquet`].
pub struct ParquetState<const N: usize> {
    mask: Array<bool, N>,
    values: Vec<Array<f64, N>>,
    dirty: Vec<[usize; N]>,
    instant: Option<Instant>,
    arena: Bump,
}

impl<const N: usize> ParquetState<N> {
    fn new(extents: [usize; N], num_fields: usize) -> Self {
        Self {
            mask: Array::zeros(extents),
            values: (0..num_fields)
                .map(|_| Array::full(extents, f64::NAN))
                .collect(),
            dirty: Vec::new(),
            instant: None,
            arena: Bump::new(),
        }
    }

    fn lend<'a>(
        signal: ArrayView<'a, bool, N>,
        fields: &'a [Array<f64, N>],
        arena: &'a mut Bump,
    ) -> (ArrayView<'a, bool, N>, &'a [ArrayView<'a, f64, N>]) {
        arena.reset();
        let fields = arena.alloc_slice_fill_iter(fields.iter().map(|f| f.view()));
        (signal, fields)
    }
}

/// Event payload for [`Parquet`].
pub struct ParquetPayload<const N: usize> {
    indices: Vec<[usize; N]>,
    columns: Vec<ArrayRef>,
}

impl<const N: usize> Source for Parquet<N> {
    type Payload = ParquetPayload<N>;
    type Instant = Instant;
    type Outputs = (SignalPort<N>, ArrayPorts<f64, N>);
    type State = ParquetState<N>;

    fn size_hint(&self) -> Option<usize> {
        // Rows in the whole file, from the footer — an upper bound when a
        // time range or a narrowing universe drops rows. On any read error
        // fall back to 0 so the aggregate stays usable.
        let count = || -> Option<usize> {
            let file = File::open(&self.path).ok()?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).ok()?;
            Some(builder.metadata().file_metadata().num_rows().max(0) as usize)
        };
        Some(count().unwrap_or(0))
    }

    fn init(
        self,
    ) -> (
        ParquetState<N>,
        impl Stream<Item = Event<ParquetPayload<N>, Instant>> + 'static,
    ) {
        let state = ParquetState::new(self.extents(), self.value_columns.len());
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || scan(&self, &tx));
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            let (tick, ts) = rx.recv().await?;
            Some((Event::at(tick, ts), rx))
        });
        (state, stream)
    }

    fn reset(
        state: &mut ParquetState<N>,
    ) -> <Self::Outputs as crate::graph::Interface>::Values<'_> {
        let mask = ArrayView::full(state.mask.extents(), &false);
        ParquetState::lend(mask, &state.values, &mut state.arena)
    }

    fn output(
        state: &mut ParquetState<N>,
    ) -> <Self::Outputs as crate::graph::Interface>::Values<'_> {
        ParquetState::lend(state.mask.view(), &state.values, &mut state.arena)
    }

    fn write(payload: ParquetPayload<N>, ts: Instant, state: &mut ParquetState<N>) -> usize {
        // Clear dirty cells in `mask` on timestamp advance.
        if state.instant.is_some_and(|last| ts > last) {
            for &index in &state.dirty {
                state.mask[index] = false;
            }
            state.dirty.clear();
        }
        state.instant = Some(ts);

        // Write data into `mask` and `values`.
        for &index in &payload.indices {
            state.mask[index] = true;
        }
        for (column, values) in payload.columns.iter().zip(state.values.iter_mut()) {
            read_value_column(column.as_ref(), &payload.indices, &mut state.mask, values);
        }

        // Update dirty list and row count.
        let num_rows_processed = payload.indices.len();
        state.dirty.extend(payload.indices);
        num_rows_processed
    }
}

/// Sequential scan on the task thread: read the table in date order and
/// send one event per distinct date per record batch.
fn scan<const N: usize>(cfg: &Parquet<N>, tx: &mpsc::Sender<(ParquetPayload<N>, Instant)>) {
    let file = File::open(&cfg.path).unwrap_or_else(|e| panic!("panel::parquet: {e}"));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .unwrap_or_else(|e| panic!("panel::parquet: {e}"));

    // Project only the columns read, so wide tables don't decode the rest.
    let schema = builder.schema();
    let mut names = vec![cfg.date_column.as_str()];
    names.extend(cfg.index_columns.iter().map(|(s, _)| s.as_str()));
    names.extend(cfg.value_columns.iter().map(|s| s.as_str()));
    let leaves: Vec<usize> = names
        .into_iter()
        .map(|name| {
            schema
                .index_of(name)
                .unwrap_or_else(|e| panic!("panel::parquet: missing column {name:?}: {e}"))
        })
        .collect();

    let mask = ProjectionMask::leaves(builder.parquet_schema(), leaves);
    let reader = builder
        .with_projection(mask)
        .build()
        .unwrap_or_else(|e| panic!("panel::parquet: {e}"));

    let index_schemas: [Axis; N] = std::array::from_fn(|j| {
        let (_, axis) = &cfg.index_columns[j];
        axis.clone()
    });

    for batch in reader {
        let batch = batch.unwrap_or_else(|e| panic!("panel::parquet: {e}"));
        if batch.num_rows() == 0 {
            continue;
        }

        let timestamps = batch
            .column_by_name(&cfg.date_column)
            .unwrap_or_else(|| panic!("panel::parquet: missing column {:?}", cfg.date_column));

        let timestamps = cast_with_options(
            timestamps.as_ref(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into())),
            &CastOptions {
                safe: false,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("panel::parquet: date column {:?}: {e}", cfg.date_column));

        let timestamps = timestamps
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();

        if timestamps.null_count() > 0 {
            panic!("panel::parquet: null in date column {:?}", cfg.date_column);
        }

        let index_columns: [ArrayRef; N] = std::array::from_fn(|j| {
            let (s, _) = &cfg.index_columns[j];
            Arc::clone(
                batch
                    .column_by_name(s)
                    .unwrap_or_else(|| panic!("panel::parquet: missing index column {s:?}")),
            )
        });

        let value_columns: Vec<ArrayRef> = cfg
            .value_columns
            .iter()
            .map(|s| {
                let column = batch
                    .column_by_name(s)
                    .unwrap_or_else(|| panic!("panel::parquet: missing value column {s:?}"));
                cast(column.as_ref(), &DataType::Float64)
                    .unwrap_or_else(|e| panic!("panel::parquet: {e}"))
            })
            .collect();

        // Split the batch into runs of equal timestamps. The table is sorted
        // by timestamp, so each run is contiguous and the runs ascend.
        let mut lo = 0;
        while lo < timestamps.len() {
            let ns = timestamps.value(lo);
            let mut hi = lo + 1;
            while hi < timestamps.len() && timestamps.value(hi) == ns {
                hi += 1;
            }
            let instant = Instant::from_offset(Duration::from_nanos(ns));
            if cfg.end.is_some_and(|e| instant >= e) {
                return;
            }
            if cfg.start.is_none_or(|s| instant >= s) {
                let sliced: [ArrayRef; N] =
                    std::array::from_fn(|j| index_columns[j].slice(lo, hi - lo));
                let indices = if N == 0 {
                    vec![[0usize; N]; hi - lo]
                } else {
                    read_index_columns(&sliced, &index_schemas)
                };
                let payload = ParquetPayload {
                    indices,
                    columns: value_columns.iter().map(|c| c.slice(lo, hi - lo)).collect(),
                };
                if tx.blocking_send((payload, instant)).is_err() {
                    return; // Receiver dropped; the run was abandoned.
                }
            }
            lo = hi;
        }
    }
}

/// Panel source from a Parquet file.
///
/// See [module-level docs](super) for inputs and outputs.
///
/// Requires a [`tokio`] runtime when added to a graph.
pub fn parquet<const N: usize>(
    path: impl Into<String>,
    index_columns: [(String, Axis); N],
    value_columns: Vec<String>,
) -> Parquet<N> {
    Parquet::new(path, index_columns, value_columns)
}
