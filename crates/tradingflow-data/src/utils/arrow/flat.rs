use arrow::array::{
    Array as ArrowArray, ArrayRef as ArrowArrayRef, PrimitiveArray as ArrowPrimitiveArray,
};
use std::sync::Arc;

use super::{ArrowScalar, as_primitive_array};
use crate::{Array, ArrayView};

/// Reads a column (Arrow array) into one-dimensional mask and value arrays.
///
/// Each `column[i]` writes `mask[i] = true` and `values[i] = column[i]` if
/// `column[i]` is non-null, and does nothing otherwise.
pub fn read_column<T: ArrowScalar>(
    column: &dyn ArrowArray,
    mask: &mut Array<bool, 1>,
    values: &mut Array<T, 1>,
) {
    let column = as_primitive_array::<T>(column);
    let [m] = values.extents();
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
    let mask = mask.data_mut();
    let values = values.data_mut();
    for (i, value) in column.iter().enumerate() {
        if let Some(value) = value {
            mask[i] = true;
            values[i] = value;
        }
    }
}

/// Builds a column (Arrow array) from one-dimensional mask and value arrays.
///
/// Each `values[[i]]` with `mask[i] == false` is written as null.
pub fn build_column<T: ArrowScalar>(
    mask: ArrayView<'_, bool, 1>,
    values: ArrayView<'_, T, 1>,
) -> ArrowArrayRef {
    assert_eq!(
        mask.extents(),
        values.extents(),
        "mask extents {:?} should match values extents {:?}",
        mask.extents(),
        values.extents(),
    );
    let column: ArrowPrimitiveArray<T::Arrow> = (mask.iter())
        .zip(values.iter())
        .map(|(&flag, &value)| flag.then_some(value))
        .collect();
    Arc::new(column)
}
