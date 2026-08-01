use std::ffi::CStr;
use std::mem::size_of;

use crate::data::Scalar;

/// An element type a [`NativeView`](super::NativeView) can export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
}

impl DType {
    /// The PEP 3118 (`struct` module) format code.
    pub const fn format(self) -> &'static CStr {
        match self {
            DType::Bool => c"?",
            DType::Int8 => c"b",
            DType::UInt8 => c"B",
            DType::Int16 => c"h",
            DType::UInt16 => c"H",
            DType::Int32 => c"i",
            DType::UInt32 => c"I",
            DType::Int64 => c"q",
            DType::UInt64 => c"Q",
            DType::Float32 => c"f",
            DType::Float64 => c"d",
        }
    }

    /// Size of one element in bytes.
    pub const fn itemsize(self) -> usize {
        match self {
            DType::Bool => size_of::<bool>(),
            DType::Int8 => size_of::<i8>(),
            DType::UInt8 => size_of::<u8>(),
            DType::Int16 => size_of::<i16>(),
            DType::UInt16 => size_of::<u16>(),
            DType::Int32 => size_of::<i32>(),
            DType::UInt32 => size_of::<u32>(),
            DType::Int64 => size_of::<i64>(),
            DType::UInt64 => size_of::<u64>(),
            DType::Float32 => size_of::<f32>(),
            DType::Float64 => size_of::<f64>(),
        }
    }

    /// The NumPy dtype name, for diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::Int8 => "int8",
            DType::UInt8 => "uint8",
            DType::Int16 => "int16",
            DType::UInt16 => "uint16",
            DType::Int32 => "int32",
            DType::UInt32 => "uint32",
            DType::Int64 => "int64",
            DType::UInt64 => "uint64",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
        }
    }
}

/// A [`Scalar`] with a matching NumPy element type, so views of it can be
/// exported to Python.
pub trait NativeScalar: Scalar {
    const DTYPE: DType;
}

impl NativeScalar for bool {
    const DTYPE: DType = DType::Bool;
}

impl NativeScalar for i8 {
    const DTYPE: DType = DType::Int8;
}

impl NativeScalar for u8 {
    const DTYPE: DType = DType::UInt8;
}

impl NativeScalar for i16 {
    const DTYPE: DType = DType::Int16;
}

impl NativeScalar for u16 {
    const DTYPE: DType = DType::UInt16;
}

impl NativeScalar for i32 {
    const DTYPE: DType = DType::Int32;
}

impl NativeScalar for u32 {
    const DTYPE: DType = DType::UInt32;
}

impl NativeScalar for i64 {
    const DTYPE: DType = DType::Int64;
}

impl NativeScalar for u64 {
    const DTYPE: DType = DType::UInt64;
}

impl NativeScalar for f32 {
    const DTYPE: DType = DType::Float32;
}

impl NativeScalar for f64 {
    const DTYPE: DType = DType::Float64;
}
