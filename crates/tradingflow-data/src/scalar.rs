//! The [`Scalar`] trait.

/// A permitted array scalar type.
pub trait Scalar: Clone + Default + Send + Sync + 'static {}

impl<T: Clone + Default + Send + Sync + 'static> Scalar for T {}
