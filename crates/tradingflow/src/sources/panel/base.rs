use arrow::array::{Array as _, ArrayRef, RecordBatch, TimestampNanosecondArray};
use arrow::compute::{CastOptions, cast, cast_with_options};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::error::ArrowError;
use bumpalo::Bump;
use futures::Stream;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::data::utils::arrow::{read_index_columns, read_value_column};
use crate::data::{Array, ArrayView, Axis, Duration, Instant};
use crate::graph::{Event, Source};
use crate::ports::{ArrayPorts, SignalPort};

/// Base trait for Arrow panel source readers.
pub trait Reader: Send + 'static {
    /// Description of this reader in debug messages.
    fn desc(&self) -> String;

    /// The estimated number of rows. Called once at build time.
    fn size_hint(&self) -> Option<usize>;

    /// Iterate over batches of rows, in ascending timestamp order. Each batch
    /// of rows must contain columns of the following names:
    ///
    /// - `time_column`, convertible to [`TimestampNanosecondArray`] by cast.
    /// - `index_columns`, readable by [`read_index_columns`] with axis schemas
    ///   specified in `index_columns`.
    /// - `value_columns`, each readable by [`read_value_column`] with scalar
    ///   type [`f64`].
    ///
    /// Returned rows should cover the time range `[start, end)` in the source.
    /// However, returning extra rows is allowed.
    fn batches(
        &self,
        time_column: &str,
        index_columns: &[(String, Axis)],
        value_columns: &[String],
        start: Option<Instant>,
        end: Option<Instant>,
    ) -> impl Iterator<Item = Result<RecordBatch, ArrowError>>;
}

/// Source signature for Arrow panel sources.
pub struct Panel<const N: usize, R: Reader> {
    reader: R,
    time_column: String,
    index_columns: [(String, Axis); N],
    value_columns: Vec<String>,
    start: Option<Instant>,
    end: Option<Instant>,
}

impl<const N: usize, R: Reader> Panel<N, R> {
    pub fn new(
        reader: R,
        time_column: impl Into<String>,
        index_columns: [(String, Axis); N],
        value_columns: Vec<String>,
    ) -> Self {
        Self {
            reader,
            time_column: time_column.into(),
            index_columns,
            value_columns,
            start: None,
            end: None,
        }
    }

    /// Restricts emission to dates in `[start, end)`. Rows before `start` are
    /// skipped instead of carried in.
    pub fn with_time_range(mut self, start: Option<Instant>, end: Option<Instant>) -> Self {
        self.start = start;
        self.end = end;
        self
    }
}

/// Event payload of Arrow panel sources (batch of rows with a same timestamp).
pub struct PanelBatch<const N: usize> {
    indices: Vec<[usize; N]>,
    columns: Vec<ArrayRef>,
}

/// Runtime state of Arrow panel sources.
pub struct PanelState<const N: usize> {
    mask: Array<bool, N>,
    values: Vec<Array<f64, N>>,
    dirty: Vec<[usize; N]>,
    instant: Option<Instant>,
    arena: Bump,
}

impl<const N: usize, R: Reader> Source for Panel<N, R> {
    type Payload = PanelBatch<N>;
    type Instant = Instant;
    type Outputs = (SignalPort<N>, ArrayPorts<f64, N>);
    type State = PanelState<N>;

    fn size_hint(&self) -> Option<usize> {
        self.reader.size_hint()
    }

    fn init(
        self,
    ) -> (
        PanelState<N>,
        impl Stream<Item = Event<PanelBatch<N>, Instant>> + 'static,
    ) {
        // Create the state.
        let extents = std::array::from_fn(|j| match &self.index_columns[j].1 {
            Axis::Labeled(schema) => schema.len(),
            Axis::Fixed(extent) => *extent,
            Axis::None => panic!("axis {j} has no known extent"),
        });
        let state = PanelState {
            mask: Array::zeros(extents),
            values: self
                .value_columns
                .iter()
                .map(|_| Array::full(extents, f64::NAN))
                .collect(),
            dirty: Vec::new(),
            instant: None,
            arena: Bump::new(),
        };

        // Create the channel stream.
        let (tx, rx) = mpsc::channel(16);
        tokio::task::spawn_blocking(move || tx_task(&self, tx));
        let stream = futures::stream::unfold(rx, |mut rx| async move {
            let (tick, ts) = rx.recv().await?;
            Some((Event::at(tick, ts), rx))
        });

        (state, stream)
    }

    fn reset(state: &mut PanelState<N>) -> (ArrayView<'_, bool, N>, &[ArrayView<'_, f64, N>]) {
        state.arena.reset();

        // Lend an all-false mask and the current values.
        let mask = ArrayView::full(state.mask.extents(), &false);
        let values = state
            .arena
            .alloc_slice_fill_iter(state.values.iter().map(|f| f.view()));

        (mask, values)
    }

    fn output(state: &mut PanelState<N>) -> (ArrayView<'_, bool, N>, &[ArrayView<'_, f64, N>]) {
        state.arena.reset();

        // Lend the current mask and values.
        let mask = state.mask.view();
        let values = state
            .arena
            .alloc_slice_fill_iter(state.values.iter().map(|f| f.view()));

        (mask, values)
    }

    fn write(payload: PanelBatch<N>, ts: Instant, state: &mut PanelState<N>) -> usize {
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

fn tx_task<const N: usize, R: Reader>(
    panel: &Panel<N, R>,
    tx: mpsc::Sender<(PanelBatch<N>, Instant)>,
) {
    let source = panel.reader.desc();
    let axes = std::array::from_fn(|j| &panel.index_columns[j].1);
    let batches = panel.reader.batches(
        &panel.time_column,
        &panel.index_columns,
        &panel.value_columns,
        panel.start,
        panel.end,
    );

    for batch in batches {
        let batch = batch.unwrap_or_else(|e| panic!("{source}: {e}"));
        if batch.num_rows() == 0 {
            continue;
        }

        let timestamps = batch
            .column_by_name(&panel.time_column)
            .unwrap_or_else(|| panic!("{source}: missing column {:?}", panel.time_column));

        let timestamps = cast_with_options(
            timestamps.as_ref(),
            &DataType::Timestamp(TimeUnit::Nanosecond, Some("+00:00".into())),
            &CastOptions {
                safe: false,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{source}: time column {:?}: {e}", panel.time_column));

        let timestamps = timestamps
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();

        if timestamps.null_count() > 0 {
            panic!("{source}: null in time column {:?}", panel.time_column);
        }

        let index_columns: [ArrayRef; N] = std::array::from_fn(|j| {
            let (s, _) = &panel.index_columns[j];
            Arc::clone(
                batch
                    .column_by_name(s)
                    .unwrap_or_else(|| panic!("{source}: missing index column {s:?}")),
            )
        });

        let value_columns: Vec<ArrayRef> = panel
            .value_columns
            .iter()
            .map(|s| {
                let column = batch
                    .column_by_name(s)
                    .unwrap_or_else(|| panic!("{source}: missing value column {s:?}"));
                cast(column.as_ref(), &DataType::Float64)
                    .unwrap_or_else(|e| panic!("{source}: value column {s:?}: {e}"))
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
            if panel.end.is_some_and(|e| instant >= e) {
                return;
            }
            if panel.start.is_none_or(|s| instant >= s) {
                let sliced: [ArrayRef; N] =
                    std::array::from_fn(|j| index_columns[j].slice(lo, hi - lo));
                let indices = if N == 0 {
                    vec![[0usize; N]; hi - lo]
                } else {
                    read_index_columns(sliced, axes)
                };
                let payload = PanelBatch {
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
