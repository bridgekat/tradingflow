#![cfg(feature = "arrow")]
//! Conversion between Arrow columns and `(mask, values)` pairs.
//!
//! The property under test throughout is that Arrow nullity and the mask say
//! the same thing, and that the value of a null cell is never consulted: every
//! target buffer here is pre-filled with a sentinel that must survive wherever
//! the source says "no datum".

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, DictionaryArray, Float64Array, Int32Array, StringArray, UInt64Array,
};
use arrow::compute::filter;
use arrow::datatypes::{Int8Type, Int32Type};
use tradingflow_data::utils::arrow::{
    build_column, build_index_columns, build_value_column, read_column, read_index_columns,
    read_value_column, true_indices,
};
use tradingflow_data::utils::{Axis, Schema};
use tradingflow_data::{Array, ArrayView};

fn f64s(column: &ArrayRef) -> Vec<Option<f64>> {
    let column = column.as_any().downcast_ref::<Float64Array>().unwrap();
    column.iter().collect()
}

// ---------------------------------------------------------------------------
// Positional: an Arrow column is a rank-1 stream
// ---------------------------------------------------------------------------

/// Null values is skipped when writing to masks and values.
#[test]
fn read_column_skip_nulls() {
    let column = Float64Array::from(vec![Some(1.0), None, None, Some(3.0)]);
    let mut mask = Array::from_parts([4], vec![false, false, true, false].into());
    let mut values = Array::from_parts([4], vec![f64::NAN, f64::NAN, 0.0, f64::NAN].into());

    read_column(&column, &mut mask, &mut values);

    assert_eq!(mask.data(), &[true, false, true, true]);
    assert_eq!(values[[0]], 1.0);
    assert!(values[[1]].is_nan());
    assert_eq!(values[[2]], 0.0);
    assert_eq!(values[[3]], 3.0);
}

/// A column with no nulls sets every mask element.
#[test]
fn read_column_without_nulls_sets_every_mask() {
    let column = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let mut mask = Array::full([3], false);
    let mut values = Array::full([3], f64::NAN);

    read_column(&column, &mut mask, &mut values);

    assert_eq!(mask.data(), &[true, true, true]);
    assert_eq!(values.data(), &[1.0, 2.0, 3.0]);
}

/// A cleared mask element becomes a null, and a set one keeps its value —
/// including a `NaN` value, which is data rather than a hole.
#[test]
fn write_column_is_the_inverse_of_read_column() {
    let mask = Array::from_parts([4], vec![true, false, true, true].into());
    let values = Array::from_parts([4], vec![1.0, 99.0, f64::NAN, 4.0].into());

    let column = build_column(mask.view(), values.view());
    let round = f64s(&column);

    assert_eq!(round[0], Some(1.0));
    assert_eq!(round[1], None, "a clear mask is a null, not 99.0");
    assert!(
        round[2].unwrap().is_nan(),
        "NaN is data, and stays non-null"
    );
    assert_eq!(round[3], Some(4.0));

    // And back again, onto a differently-filled buffer.
    let mut mask2 = Array::full([4], false);
    let mut values2 = Array::full([4], -1.0_f64);
    read_column(&column, &mut mask2, &mut values2);
    assert_eq!(mask2, mask);
    assert_eq!(values2.data()[0], 1.0);
    assert_eq!(values2.data()[1], -1.0, "the fill under the null survives");
    assert!(values2.data()[2].is_nan());
    assert_eq!(values2.data()[3], 4.0);
}

#[test]
#[should_panic(expected = "expected column of Arrow data type Float64, got Int32")]
fn read_column_rejects_a_mismatched_column_type() {
    let column = Int32Array::from(vec![1, 2]);
    let mut mask = Array::full([2], false);
    let mut values = Array::<f64, 1>::zeros([2]);
    read_column(&column, &mut mask, &mut values);
}

// ---------------------------------------------------------------------------
// Index columns ⟷ cell indices
// ---------------------------------------------------------------------------

/// A `Some` schema reads string labels; a `None` reads any integer type.
#[test]
fn resolve_indices_reads_labelled_and_numeric_axes() {
    let symbols: ArrayRef = Arc::new(StringArray::from(vec!["CCC", "AAA", "BBB"]));
    let buckets: ArrayRef = Arc::new(Int32Array::from(vec![1, 0, 1]));
    let axes = [
        Axis::Labeled(Schema::new(["AAA", "BBB", "CCC"])),
        Axis::None,
    ];

    let indices = read_index_columns(&[symbols, buckets], &axes);

    assert_eq!(indices, vec![[2, 1], [0, 0], [1, 1]]);
}

/// A low-cardinality string column comes out of Parquet dictionary-encoded,
/// so a labelled axis must read that as readily as a plain string column.
#[test]
fn resolve_indices_reads_a_dictionary_encoded_axis() {
    let symbols: DictionaryArray<Int32Type> =
        vec!["CCC", "AAA", "CCC", "BBB"].into_iter().collect();
    let axes = [Axis::Labeled(Schema::new(["AAA", "BBB", "CCC"]))];

    let indices = read_index_columns(&[Arc::new(symbols) as ArrayRef], &axes);

    assert_eq!(indices, vec![[2], [0], [2], [1]]);
}

/// Any integer key width decodes, not just the `int32` Parquet usually picks.
#[test]
fn resolve_indices_reads_any_dictionary_key_width() {
    let symbols: DictionaryArray<Int8Type> = vec!["BBB", "AAA"].into_iter().collect();
    let axes = [Axis::Labeled(Schema::new(["AAA", "BBB"]))];

    let indices = read_index_columns(&[Arc::new(symbols) as ArrayRef], &axes);

    assert_eq!(indices, vec![[1], [0]]);
}

/// The reason dictionary entries resolve lazily. Filtering rows leaves the
/// dictionary intact, so it goes on describing labels no surviving row uses —
/// and a universe narrower than the file's is the normal case, not an edge
/// one. Resolving every entry up front would fail the batch over a label
/// nothing reads.
#[test]
fn a_dictionary_may_describe_labels_outside_the_schema() {
    let symbols: DictionaryArray<Int32Type> = vec!["AAA", "BBB"].into_iter().collect();
    // Keep only the AAA row; the dictionary still carries BBB.
    let filtered = filter(&symbols, &BooleanArray::from(vec![true, false])).unwrap();
    let axes = [Axis::Labeled(Schema::new(["AAA"]))];

    let indices = read_index_columns(&[filtered], &axes);

    assert_eq!(indices, vec![[0]], "the unreferenced BBB entry is ignored");
}

/// A label a row *does* reach is still an error, and says which label.
#[test]
#[should_panic(expected = r#"label "BBB" in string index column 0 is not in the schema"#)]
fn resolve_indices_rejects_a_referenced_label_outside_the_schema() {
    let symbols: DictionaryArray<Int32Type> = vec!["AAA", "BBB"].into_iter().collect();
    let _ = read_index_columns(
        &[Arc::new(symbols) as ArrayRef],
        &[Axis::Labeled(Schema::new(["AAA"]))],
    );
}

/// The plain path reports an out-of-schema label the same way the dictionary
/// path does — the diagnosis must not depend on the encoding.
#[test]
#[should_panic(expected = r#"label "ZZZ" in string index column 0 is not in the schema"#)]
fn a_plain_string_axis_reports_an_unknown_label_alike() {
    let symbols: ArrayRef = Arc::new(StringArray::from(vec!["AAA", "ZZZ"]));
    let _ = read_index_columns(&[symbols], &[Axis::Labeled(Schema::new(["AAA"]))]);
}

#[test]
#[should_panic(expected = "null in string index column 0")]
fn resolve_indices_rejects_a_null_dictionary_key() {
    let symbols: DictionaryArray<Int32Type> = vec![Some("AAA"), None].into_iter().collect();
    let _ = read_index_columns(
        &[Arc::new(symbols) as ArrayRef],
        &[Axis::Labeled(Schema::new(["AAA"]))],
    );
}

#[test]
#[should_panic(expected = "null in numeric index column 0")]
fn resolve_indices_rejects_a_null_position() {
    let buckets: ArrayRef = Arc::new(Int32Array::from(vec![Some(0), None]));
    let _ = read_index_columns(&[buckets], &[Axis::None]);
}

/// `set_indices` walks row-major, which is the row order the whole write side
/// agrees on.
#[test]
fn set_indices_walks_the_marked_cells_row_major() {
    let mask = Array::from_parts([3, 2], vec![true, false, false, true, true, true].into());

    assert_eq!(
        true_indices(mask.view()),
        vec![[0, 0], [1, 1], [2, 0], [2, 1]]
    );
}

/// Rank 0 has exactly one cell, and the odometer must not run off it.
#[test]
fn set_indices_handles_rank_zero() {
    let marked: Vec<[usize; 0]> = true_indices(ArrayView::scalar(&true));
    assert_eq!(marked.len(), 1, "a set rank-0 mask names its one cell");
    let clear: Vec<[usize; 0]> = true_indices(ArrayView::scalar(&false));
    assert!(clear.is_empty());
}

/// An empty axis yields no cells rather than panicking.
#[test]
fn set_indices_handles_an_empty_axis() {
    let mask = Array::<bool, 2>::zeros([0, 3]);
    assert!(true_indices(mask.view()).is_empty());
}

// ---------------------------------------------------------------------------
// Indexed: a long-format column scatters into a rank-N cross-section
// ---------------------------------------------------------------------------

/// The pivot: each row lands in the cell its index columns name, a null row
/// lands nowhere at all, and cells no row names keep whatever they held.
#[test]
fn scatter_pivots_long_rows_into_a_cross_section() {
    let symbols: ArrayRef = Arc::new(StringArray::from(vec!["AAA", "AAA", "CCC", "BBB"]));
    let buckets: ArrayRef = Arc::new(UInt64Array::from(vec![0_u64, 1, 1, 0]));
    let axes = [
        Axis::Labeled(Schema::new(["AAA", "BBB", "CCC"])),
        Axis::None,
    ];
    let indices = read_index_columns(&[symbols, buckets], &axes);
    // The BBB row is null: it must leave both the mask and the fill alone.
    let column = Float64Array::from(vec![Some(1.0), Some(2.0), Some(5.0), None]);

    let mut mask = Array::full([3, 2], false);
    let mut values = Array::full([3, 2], f64::NAN);
    read_value_column(&column, &indices, &mut mask, &mut values);

    assert_eq!(
        mask.data(),
        &[true, true, false, false, false, true],
        "only the three non-null rows are marked"
    );
    assert_eq!(values[[0, 0]], 1.0);
    assert_eq!(values[[0, 1]], 2.0);
    assert_eq!(values[[2, 1]], 5.0);
    for cell in [[1, 0], [1, 1], [2, 0]] {
        assert!(values[cell].is_nan(), "cell {cell:?} keeps its fill");
    }
}

/// The `_into` contract: a scatter names the cells it writes and touches
/// nothing else, so a source can carry a cross-section across ticks and clear
/// on its own schedule.
#[test]
fn scatter_leaves_unnamed_cells_untouched() {
    let indices = vec![[1_usize]];
    let column = Float64Array::from(vec![7.0]);

    let mut mask = Array::from_parts([3], vec![true, false, true].into());
    let mut values = Array::from_parts([3], vec![10.0, 20.0, 30.0].into());
    read_value_column(&column, &indices, &mut mask, &mut values);

    assert_eq!(mask.data(), &[true, true, true], "no mask is cleared");
    assert_eq!(values.data(), &[10.0, 7.0, 30.0]);
}

/// Fields of one table share their index columns, so they share one
/// resolution — and land on identical cells.
#[test]
fn one_resolution_serves_every_field_of_a_batch() {
    let symbols: ArrayRef = Arc::new(StringArray::from(vec!["BBB", "AAA"]));
    let axes = [Axis::Labeled(Schema::new(["AAA", "BBB"]))];
    let indices = read_index_columns(&[symbols], &axes);

    let mut mask = Array::full([2], false);
    let mut opens = Array::full([2], f64::NAN);
    let mut volumes = Array::<u64, 1>::zeros([2]);
    read_value_column(
        &Float64Array::from(vec![1.5, 2.5]),
        &indices,
        &mut mask,
        &mut opens,
    );
    // A second field of the same batch, with its own scalar type.
    read_value_column(
        &UInt64Array::from(vec![100_u64, 200]),
        &indices,
        &mut mask,
        &mut volumes,
    );

    assert_eq!(mask.data(), &[true, true]);
    assert_eq!(opens.data(), &[2.5, 1.5]);
    assert_eq!(volumes.data(), &[200, 100]);
}

#[test]
#[should_panic(expected = "out of bounds for extents [2, 2]")]
fn scatter_rejects_an_out_of_bounds_cell() {
    let indices = vec![[0_usize, 5]];
    let column = Float64Array::from(vec![1.0]);
    let mut mask = Array::full([2, 2], false);
    let mut values = Array::<f64, 2>::zeros([2, 2]);
    read_value_column(&column, &indices, &mut mask, &mut values);
}

#[test]
#[should_panic(expected = "column should have exactly 1 elements, got 2")]
fn scatter_rejects_a_row_count_mismatch() {
    let indices = vec![[0_usize]];
    let column = Float64Array::from(vec![1.0, 2.0]);
    let mut mask = Array::full([2], false);
    let mut values = Array::<f64, 1>::zeros([2]);
    read_value_column(&column, &indices, &mut mask, &mut values);
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

/// The whole pivot, out and back: a cross-section written as a long table and
/// read into a fresh cross-section is the one we started with. A clear mask
/// produces no row rather than a null one, so the unmarked cells come back as
/// the new buffer's fill.
#[test]
fn a_cross_section_round_trips_through_a_long_table() {
    let axes = [
        Axis::Labeled(Schema::new(["AAA", "BBB", "CCC"])),
        Axis::None,
    ];
    let mask = Array::from_parts([3, 2], vec![true, true, false, false, false, true].into());
    let values = Array::from_parts([3, 2], vec![1.0, 2.0, 9.0, 9.0, 9.0, 5.0].into());

    // Out: the marked cells become rows.
    let indices = true_indices(mask.view());
    let [symbols, buckets] = build_index_columns(&indices, &axes);
    let column = build_value_column(&indices, mask.view(), values.view());

    // Back: the rows land on the cells they came from.
    let round = read_index_columns(&[symbols, buckets], &axes);
    assert_eq!(round, indices);

    let mut mask2 = Array::full([3, 2], false);
    let mut values2 = Array::full([3, 2], f64::NAN);
    read_value_column(&column, &round, &mut mask2, &mut values2);

    assert_eq!(mask2, mask);
    for cell in [[0, 0], [0, 1], [2, 1]] {
        assert_eq!(values2[cell], values[cell], "cell {cell:?}");
    }
    for cell in [[1, 0], [1, 1], [2, 0]] {
        assert!(
            values2[cell].is_nan(),
            "unmarked cell {cell:?} carries no row, so it keeps the new fill"
        );
    }
}
