use arrow::array::{Array as _, RecordBatch};
use arrow::error::ArrowError;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ParquetRecordBatchReaderBuilder};
use std::fs::File;

use super::base::{Panel, Reader, cast_timestamps};
use crate::data::utils::Axis;
use crate::data::{Duration, Instant};

/// Arrow panel source reader for [`parquet()`].
pub struct ParquetReader {
    path: String,
}

impl ParquetReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

/// A row group overlapping the requested time range.
struct Selected {
    /// Its index in the file.
    group: usize,
    /// Whether the range cuts through it, so that only some of its rows are
    /// emitted and its row count cannot be read off the metadata.
    clipped: bool,
}

/// The row groups whose time-column statistics overlap `[start, end)`, in file
/// order, or `None` when there is no range to apply or the statistics cannot be
/// read. A group whose bounds are missing is kept and marked clipped, so the
/// selection never drops rows and never overstates a count.
fn select(
    metadata: &ArrowReaderMetadata,
    time_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
) -> Option<Vec<Selected>> {
    if start.is_none() && end.is_none() {
        return None;
    }

    let groups = metadata.metadata().row_groups();
    let converter =
        StatisticsConverter::try_new(time_column, metadata.schema(), metadata.parquet_schema())
            .ok()?;
    let mins = cast_timestamps(converter.row_group_mins(groups.iter()).ok()?.as_ref()).ok()?;
    let maxes = cast_timestamps(converter.row_group_maxes(groups.iter()).ok()?.as_ref()).ok()?;

    let instant = |ns| Instant::from_offset(Duration::from_nanos(ns));
    let min = |i: usize| mins.is_valid(i).then(|| instant(mins.value(i)));
    let max = |i: usize| maxes.is_valid(i).then(|| instant(maxes.value(i)));

    Some(
        (0..groups.len())
            .filter_map(|group| {
                // A group holds times in `[min, max]`, so it misses the range
                // only by ending before `start` or beginning at or after `end`.
                let before = max(group).is_some_and(|t| start.is_some_and(|s| t < s));
                let after = min(group).is_some_and(|t| end.is_some_and(|e| t >= e));
                if before || after {
                    return None;
                }
                // It contributes all of its rows only if both bounds are known
                // and both lie inside the range.
                let from = min(group).is_some_and(|t| start.is_none_or(|s| t >= s));
                let to = max(group).is_some_and(|t| end.is_none_or(|e| t < e));
                Some(Selected {
                    group,
                    clipped: !(from && to),
                })
            })
            .collect(),
    )
}

/// The rows of `group` whose timestamps fall in `[start, end)`, by reading that
/// group's time column alone. Cheap next to a full scan: one column of one
/// group, and only the groups the range cuts through are ever read this way.
fn count(
    file: File,
    metadata: &ArrowReaderMetadata,
    group: usize,
    time_column: &str,
    start: Option<Instant>,
    end: Option<Instant>,
) -> Option<usize> {
    let index = metadata.schema().index_of(time_column).ok()?;
    let proj = ProjectionMask::leaves(metadata.parquet_schema(), [index]);
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
        .with_row_groups(vec![group])
        .with_projection(proj)
        .build()
        .ok()?;

    let mut rows = 0;
    for batch in reader {
        let batch = batch.ok()?;
        let column = batch.column_by_name(time_column)?;
        let timestamps = cast_timestamps(column.as_ref()).ok()?;
        if timestamps.null_count() > 0 {
            return None;
        }
        // The table is sorted by timestamp, so the rows in range form a run.
        let timestamps = Instant::from_nanos_slice(timestamps.values());
        let lo = start.map_or(0, |s| timestamps.partition_point(|&t| t < s));
        let hi = end.map_or(timestamps.len(), |e| timestamps.partition_point(|&t| t < e));
        rows += hi.saturating_sub(lo);
    }
    Some(rows)
}

impl Reader for ParquetReader {
    fn desc(&self) -> String {
        format!("panel::parquet(\"{}\")", self.path)
    }

    fn size_hint(
        &self,
        time_column: &str,
        start: Option<Instant>,
        end: Option<Instant>,
    ) -> Option<usize> {
        // Without a time range the footer already carries the answer. With one,
        // add up the row groups the range covers: from the metadata for a group
        // lying wholly inside it, and from the time column for the two the range
        // cuts through. On any read error fall back to the whole file, an upper
        // bound, so the aggregate stays usable.
        let rows = || -> Option<usize> {
            let file = File::open(&self.path).ok()?;
            let metadata = ArrowReaderMetadata::load(&file, Default::default()).ok()?;

            let Some(selected) = select(&metadata, time_column, start, end) else {
                return Some(metadata.metadata().file_metadata().num_rows().max(0) as usize);
            };

            let mut rows = 0;
            for Selected { group, clipped } in selected {
                let whole = metadata.metadata().row_group(group).num_rows().max(0) as usize;
                rows += if clipped {
                    let file = File::open(&self.path).ok()?;
                    // An unreadable group falls back to its full size, which
                    // keeps the estimate an upper bound.
                    count(file, &metadata, group, time_column, start, end).unwrap_or(whole)
                } else {
                    whole
                };
            }
            Some(rows)
        };
        Some(rows().unwrap_or(0))
    }

    fn batches(
        &self,
        time_column: &str,
        index_columns: &[(String, Axis)],
        value_columns: &[String],
        start: Option<Instant>,
        end: Option<Instant>,
    ) -> impl Iterator<Item = Result<RecordBatch, ArrowError>> {
        let source = self.desc();
        let file = File::open(&self.path).unwrap_or_else(|e| panic!("{source}: {e}"));
        let metadata = ArrowReaderMetadata::load(&file, Default::default())
            .unwrap_or_else(|e| panic!("{source}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone());

        // Project only the columns read, so wide tables don't decode the rest.
        let schema = builder.schema();
        let columns = std::iter::once(time_column)
            .chain(index_columns.iter().map(|(s, _)| s.as_str()))
            .chain(value_columns.iter().map(|s| s.as_str()));
        let leaves: Vec<usize> = columns
            .map(|name| {
                schema
                    .index_of(name)
                    .unwrap_or_else(|e| panic!("{source}: missing column {name:?}: {e}"))
            })
            .collect();
        let proj = ProjectionMask::leaves(builder.parquet_schema(), leaves);

        // Skip the row groups the time range misses, so that a window into a
        // long history does not decode the years before it. A group the range
        // only clips is read whole and trimmed downstream, which the reader
        // contract allows.
        let mut builder = builder.with_projection(proj);
        if let Some(selected) = select(&metadata, time_column, start, end) {
            builder = builder.with_row_groups(selected.iter().map(|s| s.group).collect());
        }
        builder.build().unwrap_or_else(|e| panic!("{source}: {e}"))
    }
}

/// Panel source from a Parquet file.
///
/// See [module-level docs](super) for inputs and outputs.
///
/// Requires a [`tokio`] runtime when added to a graph.
pub fn parquet<const N: usize>(
    path: impl Into<String>,
    time_column: impl Into<String>,
    index_columns: [(String, Axis); N],
    value_columns: Vec<String>,
) -> Panel<N, ParquetReader> {
    Panel::new(
        ParquetReader::new(path),
        time_column,
        index_columns,
        value_columns,
    )
}
