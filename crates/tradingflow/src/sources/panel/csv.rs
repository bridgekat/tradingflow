use arrow::array::{Array as _, RecordBatch};
use arrow::csv::ReaderBuilder;
use arrow::csv::reader::Format;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;

use super::base::{Panel, Reader, cast_timestamps};
use crate::data::utils::Axis;
use crate::data::{Duration, Instant};

/// Rows sniffed to type the time column. Only that column's type is inferred
/// — the others are pinned by the axes and by the `f64` value type — so a
/// sample is enough unless the column's format changes mid-file.
const INFER_ROWS: usize = 1024;

/// Bytes per read when counting rows for [`Reader::size_hint`].
const COUNT_CHUNK: usize = 64 * 1024;

/// Bytes per read when scanning for a row boundary. Rows are short, so a
/// bisection probe rarely needs more than one.
const PROBE_CHUNK: usize = 8 * 1024;

/// Arrow panel source reader for [`csv()`].
pub struct CsvReader {
    path: String,
}

impl CsvReader {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// Sniffs the file's format and schema from its header and leading rows.
    fn infer(&self) -> Result<(Format, Schema), ArrowError> {
        let format = Format::default().with_header(true);
        let file = File::open(&self.path)?;
        let (schema, _) = format.infer_schema(BufReader::new(file), Some(INFER_ROWS))?;
        Ok((format, schema))
    }

    /// The header row, and the byte range holding the rows whose timestamps
    /// fall in `[start, end)` — the whole file when there is no time range.
    ///
    /// The table is sorted by timestamp, so those rows are contiguous and their
    /// bounds can be found by bisecting over byte offsets, at a seek and a
    /// single row's worth of parsing per probe. Both offsets land on a row
    /// boundary, so the range can be counted and read as it stands, once the
    /// header is put back in front of it.
    fn span(
        &self,
        file: &mut File,
        len: u64,
        time_column: &str,
        start: Option<Instant>,
        end: Option<Instant>,
    ) -> Option<(Vec<u8>, u64, u64)> {
        let header = read_row(file, 0)?;
        let data = header.len() as u64;
        if start.is_none() && end.is_none() {
            return Some((header, data, len));
        }

        let (format, schema) = self.infer().ok()?;
        let probe = Probe {
            header: &header,
            time: schema.index_of(time_column).ok()?,
            schema: Arc::new(schema),
            format,
        };

        let from = match start {
            Some(t) => probe.lower_bound(file, len, data, t)?,
            None => data,
        };
        let to = match end {
            Some(t) => probe.lower_bound(file, len, data, t)?,
            None => len,
        };

        // An unsorted table can bisect to an inverted range: read nothing
        // rather than a negative span, and let the reader reject the order.
        Some((header, from, to.max(from)))
    }
}

/// What a bisection probe needs to read one row's timestamp.
struct Probe<'a> {
    /// The header row, put in front of the probed row so that Arrow can parse
    /// it the same way it parses the table.
    header: &'a [u8],
    format: Format,
    schema: Arc<Schema>,
    /// The index of the time column in `schema`.
    time: usize,
}

impl Probe<'_> {
    /// The timestamp of the row starting at `at`.
    fn instant(&self, file: &mut File, at: u64) -> Option<Instant> {
        let mut bytes = self.header.to_vec();
        bytes.extend_from_slice(&read_row(file, at)?);
        if !bytes.ends_with(b"\n") {
            bytes.push(b'\n');
        }

        let mut reader = ReaderBuilder::new(Arc::clone(&self.schema))
            .with_format(self.format.clone())
            .with_projection(vec![self.time])
            .with_batch_size(1)
            .build(Cursor::new(bytes))
            .ok()?;

        let batch = reader.next()?.ok()?;
        let timestamps = cast_timestamps(batch.column(0).as_ref()).ok()?;
        (!timestamps.is_empty() && timestamps.is_valid(0))
            .then(|| Instant::from_offset(Duration::from_nanos(timestamps.value(0))))
    }

    /// The offset of the first row whose timestamp is at or after `t`, or the
    /// end of the file when there is none. Rows before `data` are the header.
    fn lower_bound(&self, file: &mut File, len: u64, data: u64, t: Instant) -> Option<u64> {
        let (mut lo, mut hi) = (data, len);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let reached = match row_start(file, len, data, mid) {
                // A probe past the last row counts as at or after any
                // timestamp, which keeps the predicate monotone.
                None => true,
                Some(start) => self.instant(file, start)? >= t,
            };
            if reached { hi = mid } else { lo = mid + 1 }
        }
        // `lo` is the earliest offset that probes the row looked for, which is
        // one past the start of the row before it rather than the boundary
        // itself. Map it back onto the row it probes.
        Some(row_start(file, len, data, lo).unwrap_or(len))
    }
}

/// The bytes of the row starting at `at`, up to and including its line break.
fn read_row(file: &mut File, at: u64) -> Option<Vec<u8>> {
    file.seek(SeekFrom::Start(at)).ok()?;
    let mut row = Vec::new();
    let mut buf = vec![0u8; PROBE_CHUNK];
    loop {
        let read = file.read(&mut buf).ok()?;
        if read == 0 {
            return Some(row); // A last row left unterminated is still a row.
        }
        match buf[..read].iter().position(|&b| b == b'\n') {
            Some(i) => {
                row.extend_from_slice(&buf[..=i]);
                return Some(row);
            }
            None => row.extend_from_slice(&buf[..read]),
        }
    }
}

/// The offset of the first row that starts at or after `at`, or `None` when the
/// file holds no such row.
fn row_start(file: &mut File, len: u64, data: u64, at: u64) -> Option<u64> {
    if at <= data {
        return (data < len).then_some(data);
    }
    if at > len {
        return None;
    }

    // A row starts just past a line break, so search from `at - 1`: an offset
    // already on a row boundary then maps to itself, and every other offset
    // within a row maps to the row after it.
    let mut pos = at - 1;
    file.seek(SeekFrom::Start(pos)).ok()?;
    let mut buf = vec![0u8; PROBE_CHUNK];
    loop {
        let read = file.read(&mut buf).ok()?;
        if read == 0 {
            return None;
        }
        if let Some(i) = buf[..read].iter().position(|&b| b == b'\n') {
            let start = pos + i as u64 + 1;
            return (start < len).then_some(start);
        }
        pos += read as u64;
    }
}

/// The number of rows in `[from, to)`, both of which lie on a row boundary.
fn count_rows(file: &mut File, len: u64, from: u64, to: u64) -> Option<usize> {
    if to <= from {
        return Some(0);
    }
    file.seek(SeekFrom::Start(from)).ok()?;

    let mut reader = BufReader::with_capacity(COUNT_CHUNK, file.by_ref().take(to - from));
    let mut rows = 0usize;
    let mut last = None;
    loop {
        let buf = reader.fill_buf().ok()?;
        if buf.is_empty() {
            // A last row left unterminated is still a row.
            rows += usize::from(to == len && last.is_some_and(|b| b != b'\n'));
            return Some(rows);
        }
        rows += buf.iter().filter(|&&b| b == b'\n').count();
        last = buf.last().copied();
        let read = buf.len();
        reader.consume(read);
    }
}

impl Reader for CsvReader {
    fn desc(&self) -> String {
        format!("panel::csv(\"{}\")", self.path)
    }

    fn size_hint(
        &self,
        time_column: &str,
        start: Option<Instant>,
        end: Option<Instant>,
    ) -> Option<usize> {
        // A CSV file carries no row count, so count the line breaks over the
        // byte range holding the requested rows — the whole file, less the
        // header, when there is no time range. Exact but for a quoted line
        // break inside a field, and cheaper than a full scan whenever the range
        // is a slice of the table. On any read error fall back to 0 so the
        // aggregate stays usable.
        let rows = || -> Option<usize> {
            let mut file = File::open(&self.path).ok()?;
            let len = file.metadata().ok()?.len();
            let (_, from, to) = self.span(&mut file, len, time_column, start, end)?;
            count_rows(&mut file, len, from, to)
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

        // A CSV file carries no schema, so sniff one from the header and the
        // leading rows. Only the time column keeps its inferred type: an index
        // column is read as its axis demands (labels as strings, indices as
        // integers) and a value column as `f64`, so a label that happens to look
        // numeric — a bare numeric symbol, say — is still read as a label.
        let (format, inferred) = self.infer().unwrap_or_else(|e| panic!("{source}: {e}"));

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

        // Read only the byte range the time range covers, so a window into a
        // long history does not parse the years before it. Should the search
        // fail, fall back to the whole file behind an empty header: reading
        // rows outside the range only costs time, which the contract allows.
        let mut file = File::open(&self.path).unwrap_or_else(|e| panic!("{source}: {e}"));
        let len = file
            .metadata()
            .unwrap_or_else(|e| panic!("{source}: {e}"))
            .len();
        let (header, from, to) = self
            .span(&mut file, len, time_column, start, end)
            .unwrap_or((Vec::new(), 0, len));
        file.seek(SeekFrom::Start(from))
            .unwrap_or_else(|e| panic!("{source}: {e}"));
        let rows = Cursor::new(header).chain(file.take(to - from));

        ReaderBuilder::new(Arc::new(Schema::new(fields)))
            .with_format(format)
            .with_projection(proj)
            .build(BufReader::new(rows))
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
