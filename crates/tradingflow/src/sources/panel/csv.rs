use arrow::array::RecordBatch;
use arrow::csv::ReaderBuilder;
use arrow::csv::reader::Format;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

use super::base::{Panel, Reader};
use crate::data::Instant;
use crate::data::utils::Axis;

/// Rows sniffed to type the time column. Only that column's type is inferred
/// — the others are pinned by the axes and by the `f64` value type — so a
/// sample is enough unless the column's format changes mid-file.
const INFER_ROWS: usize = 1024;

/// Bytes per read when counting rows for [`Reader::size_hint`].
const COUNT_CHUNK: usize = 64 * 1024;

/// Arrow panel source reader for [`csv()`].
pub struct CsvReader {
    path: String,
}

impl CsvReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

impl Reader for CsvReader {
    fn desc(&self) -> String {
        format!("panel::csv(\"{}\")", self.path)
    }

    fn size_hint(&self) -> Option<usize> {
        // A CSV file carries no row count, so count its lines — an upper
        // bound, since a time range drops rows and a quoted newline inside a
        // field adds one. On any read error fall back to 0 so the aggregate
        // stays usable.
        let count = || -> Option<usize> {
            let file = File::open(&self.path).ok()?;
            let mut reader = BufReader::with_capacity(COUNT_CHUNK, file);
            let mut lines = 0usize;
            let mut last = None;
            loop {
                let buf = reader.fill_buf().ok()?;
                if buf.is_empty() {
                    // A last row left unterminated is still a row.
                    lines += usize::from(last.is_some_and(|b| b != b'\n'));
                    return Some(lines.saturating_sub(1)); // Less the header.
                }
                lines += buf.iter().filter(|&&b| b == b'\n').count();
                last = buf.last().copied();
                let read = buf.len();
                reader.consume(read);
            }
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
        let open = || File::open(&self.path).unwrap_or_else(|e| panic!("{source}: {e}"));

        // A CSV file carries no schema, so sniff one from the header and the
        // leading rows. Only the time column keeps its inferred type: an index
        // column is read as its axis demands (labels as strings, indices as
        // integers) and a value column as `f64`, so a label that happens to look
        // numeric — a bare numeric symbol, say — is still read as a label.
        let format = Format::default().with_header(true);
        let (inferred, _) = format
            .infer_schema(BufReader::new(open()), Some(INFER_ROWS))
            .unwrap_or_else(|e| panic!("{source}: {e}"));

        let mut fields: Vec<Arc<Field>> = inferred.fields().iter().map(Arc::clone).collect();
        let mut typed = |name: &str, ty: Option<DataType>| -> usize {
            let index = inferred
                .index_of(name)
                .unwrap_or_else(|e| panic!("{source}: missing column {name:?}: {e}"));
            if let Some(ty) = ty {
                fields[index] = Arc::new(Field::new(name, ty, true));
            }
            index
        };

        // Project only the columns read, so wide tables don't parse the rest.
        let mut proj = vec![typed(time_column, None)];
        for (name, axis) in index_columns {
            let ty = match axis {
                Axis::Labeled(_) => DataType::Utf8,
                _ => DataType::Int64,
            };
            proj.push(typed(name, Some(ty)));
        }
        for name in value_columns {
            proj.push(typed(name, Some(DataType::Float64)));
        }
        proj.sort_unstable();
        proj.dedup();

        ReaderBuilder::new(Arc::new(Schema::new(fields)))
            .with_format(format)
            .with_projection(proj)
            .build(BufReader::new(open()))
            .unwrap_or_else(|e| panic!("{source}: {e}"))
    }
}

/// Panel source from a CSV file.
///
/// See [module-level docs](super) for inputs and outputs.
///
/// Requires a [`tokio`] runtime when added to a graph.
///
/// # Column types
///
/// Compared to Parquet, a CSV file does not carry its own schema, so the
/// source assigns the types itself. The file must have a header row, which
/// names the columns. Index and value columns are typed from the request —
/// labeled axes parse as strings, numeric axes as integers, value columns
/// as `f64` — so a label that looks numeric is still read as a label.
///
/// The time column's type is sniffed, from the first 1024 rows, which
/// admits calendar dates (`2024-01-02`), timestamps (`2024-01-02 09:30:00`)
/// and integer epoch nanoseconds.
///
/// In all columns, empty cells are considered as nulls.
pub fn csv<const N: usize>(
    path: impl Into<String>,
    time_column: impl Into<String>,
    index_columns: [(String, Axis); N],
    value_columns: Vec<String>,
) -> Panel<N, CsvReader> {
    Panel::new(
        CsvReader::new(path),
        time_column,
        index_columns,
        value_columns,
    )
}
