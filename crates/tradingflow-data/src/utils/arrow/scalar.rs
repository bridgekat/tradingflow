use arrow::array::{Array as ArrowArray, PrimitiveArray as ArrowPrimitiveArray};
use arrow::datatypes::{
    ArrowNativeType, ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

use crate::Scalar;

/// A [`Scalar`] that is the native representation of an Arrow primitive type.
///
/// Only these are supported for conversion now.
pub trait ArrowScalar: Scalar + ArrowNativeType {
    type Arrow: ArrowPrimitiveType<Native = Self>;
}

macro_rules! impl_arrow_scalar {
    ($($native:ty => $arrow:ty),* $(,)?) => {
        $( impl ArrowScalar for $native { type Arrow = $arrow; } )*
    };
}

impl_arrow_scalar! {
    i8 => Int8Type,
    i16 => Int16Type,
    i32 => Int32Type,
    i64 => Int64Type,
    u8 => UInt8Type,
    u16 => UInt16Type,
    u32 => UInt32Type,
    u64 => UInt64Type,
    f32 => Float32Type,
    f64 => Float64Type,
}

/// Downcasts a column to the primitive array matching `T`.
///
/// # Panics
///
/// Panics if the column has any other Arrow data type.
pub fn as_primitive_array<T: ArrowScalar>(
    column: &dyn ArrowArray,
) -> &ArrowPrimitiveArray<T::Arrow> {
    column
        .as_any()
        .downcast_ref::<ArrowPrimitiveArray<T::Arrow>>()
        .unwrap_or_else(|| {
            panic!(
                "expected column of Arrow data type {}, got {}",
                T::Arrow::DATA_TYPE,
                column.data_type(),
            )
        })
}
