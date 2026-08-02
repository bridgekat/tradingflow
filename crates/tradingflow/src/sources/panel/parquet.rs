use arrow::array::RecordBatch;
use arrow::error::ArrowError;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

use super::base::{Panel, Reader};
use crate::data::Instant;
use crate::data::utils::Axis;

/// Arrow panel source reader for [`parquet()`].
pub struct ParquetReader {
    path: String,
}

impl ParquetReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl Reader for ParquetReader {
    fn desc(&self) -> String {
        format!("panel::parquet(\"{}\")", self.path)
    }

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

    fn batches(
        &self,
        time_column: &str,
        index_columns: &[(String, Axis)],
        value_columns: &[String],
        _start: Option<Instant>,
        _end: Option<Instant>,
    ) -> impl Iterator<Item = Result<RecordBatch, ArrowError>> {
        let source = self.desc();
        let file = File::open(&self.path).unwrap_or_else(|e| panic!("{source}: {e}"));
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap_or_else(|e| panic!("{source}: {e}"));

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

        builder
            .with_projection(proj)
            .build()
            .unwrap_or_else(|e| panic!("{source}: {e}"))
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
