use arrow::array::{
    Array as ArrowArray, ArrayRef as ArrowArrayRef, DictionaryArray as ArrowDictionaryArray,
    PrimitiveArray as ArrowPrimitiveArray, StringArray as ArrowStringArray,
    UInt64Array as ArrowUInt64Array, as_string_array, downcast_dictionary_array,
    downcast_integer_array,
};
use arrow::datatypes::{ArrowNativeType, DataType, UInt64Type};
use std::sync::Arc;

use super::{ArrowScalar, as_primitive_array};
use crate::{Array, ArrayView, Axis, Layout};

/// Reads a column (Arrow array) into locations specified by `indices` in
/// `N`-dimensional mask and value arrays,
///
/// Each pair `(indices[i], column[i])` writes `mask[indices[i]] = true`
/// and `values[indices[i]] = column[i]` if `column[i]` is non-null, and does
/// nothing otherwise.
pub fn read_value_column<T: ArrowScalar, const N: usize>(
    column: &dyn ArrowArray,
    indices: &[[usize; N]],
    mask: &mut Array<bool, N>,
    values: &mut Array<T, N>,
) {
    let column = as_primitive_array::<T>(column);
    let m = indices.len();
    assert_eq!(
        column.len(),
        m,
        "column should have exactly {} elements, got {}",
        m,
        column.len(),
    );
    assert_eq!(
        mask.extents(),
        values.extents(),
        "mask extents {:?} should match values extents {:?}",
        mask.extents(),
        values.extents(),
    );
    for (&index, value) in indices.iter().zip(column.iter()) {
        if let Some(value) = value {
            mask[index] = true;
            values[index] = value;
        }
    }
}

/// Builds a column (Arrow array) from locations specified by `indices`
/// in `N`-dimensional mask and value arrays,
///
/// Each `values[index]` with `mask[index] == false` is written as null.
pub fn build_value_column<T: ArrowScalar, const N: usize>(
    indices: &[[usize; N]],
    mask: ArrayView<'_, bool, N>,
    values: ArrayView<'_, T, N>,
) -> ArrowArrayRef {
    assert_eq!(
        mask.extents(),
        values.extents(),
        "mask extents {:?} should match values extents {:?}",
        mask.extents(),
        values.extents(),
    );
    let column: ArrowPrimitiveArray<T::Arrow> = (indices.iter())
        .map(|&index| mask[index].then_some(values[index]))
        .collect();
    Arc::new(column)
}

/// Reads `N` index columns (Arrow arrays) into `N`-dimensional indices.
///
/// At least one index column is required; use `vec![[_; 0]; len]` otherwise.
///
/// An axis with [`Axis::Labeled`] should correspond to a string-valued column,
/// which can contain either plain or dictionary-encoded strings. In all other
/// cases, the axis should correspond to an integer-valued column.
pub fn read_index_columns<const N: usize>(
    columns: [ArrowArrayRef; N],
    axes: [&Axis; N],
) -> Vec<[usize; N]> {
    let m = columns
        .first()
        .unwrap_or_else(|| panic!("function expects at least one index column"))
        .len();
    for (j, column) in columns.iter().enumerate() {
        assert_eq!(
            column.len(),
            m,
            "index column {j} should have exactly {m} elements, got {}",
            column.len(),
        );
    }
    let mut res = vec![[0usize; N]; m];
    for (j, (column, axis)) in columns.iter().zip(axes.iter()).enumerate() {
        if let Axis::Labeled(schema) = axis {
            downcast_dictionary_array!(
                column => {
                    let palette = as_string_array(column.values());
                    let palette_with_index: Vec<Option<(&str, Option<usize>)>> = palette
                        .iter()
                        .map(|value| value.map(|name| (name, schema.try_index(name))))
                        .collect();
                    for (i, entry) in column.keys().iter().enumerate() {
                        let key = entry.unwrap_or_else(|| panic!("null in string index column {j}")).as_usize();
                        let (name, index) = palette_with_index[key]
                            .unwrap_or_else(|| panic!("null in string index column {j}"));
                        res[i][j] = index
                            .unwrap_or_else(|| panic!("label {name:?} in string index column {j} is not in the schema"));
                    }
                }
                DataType::Utf8 => {
                    for (i, entry) in as_string_array(column).iter().enumerate() {
                        let name = entry.unwrap_or_else(|| panic!("null in string index column {j}"));
                        res[i][j] = schema
                            .try_index(name)
                            .unwrap_or_else(|| panic!("label {name:?} in string index column {j} is not in the schema"));
                    }
                }
                ty => panic!("unsupported Arrow data type {ty} for named index")
            );
        } else {
            downcast_integer_array!(
                column => {
                    for (i, entry) in column.iter().enumerate() {
                        let Some(index) = entry else {
                            panic!("null in numeric index column {j}")
                        };
                        res[i][j] = index.as_usize();
                    }
                }
                ty => panic!("unsupported Arrow data type {ty} for numeric index")
            );
        }
    }
    res
}

/// Converts `N`-dimensional indices into `N` index columns (Arrow arrays).
pub fn build_index_columns<const N: usize>(
    indices: &[[usize; N]],
    axes: [&Axis; N],
) -> [ArrowArrayRef; N] {
    std::array::from_fn(|j| {
        if let Axis::Labeled(schema) = &axes[j] {
            let palette = ArrowStringArray::from_iter_values(schema.labels().iter());
            let keys =
                ArrowUInt64Array::from_iter_values(indices.iter().map(|index| index[j] as u64));
            let column =
                ArrowDictionaryArray::<UInt64Type>::try_new(keys, Arc::new(palette)).unwrap();
            Arc::new(column) as ArrowArrayRef
        } else {
            let column =
                ArrowUInt64Array::from_iter_values(indices.iter().map(|index| index[j] as u64));
            Arc::new(column) as ArrowArrayRef
        }
    })
}

/// Returns the list of indices of `true` elements, in row-major order.
pub fn true_indices<const N: usize>(mask: ArrayView<'_, bool, N>) -> Vec<[usize; N]> {
    mask.layout()
        .iter()
        .with_indices()
        .filter_map(|(index, offset)| mask.data()[offset].then_some(index))
        .collect()
}
