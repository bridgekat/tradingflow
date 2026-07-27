//! [`PyArgs`] — build the Python view tuple + produced bools from typed input
//! refs.

use pyo3::prelude::*;

use crate::graph::Interface;
use crate::graph::typed::InterfaceHandles;
use crate::ports::{ArrayPort, ArrayPorts, ClockPort, SeriesPort, SeriesPorts, is_eventful};

use super::{NativeArrayView, NativeSeriesView};

/// Walks an operator's input [`Interface`] refs tree, appending one Python view
/// per leaf ([`NativeArrayView`] / [`NativeSeriesView`] / `None`) and one
/// produced (notify) bool per leaf, in tree order.
pub trait PyArgs: Interface + InterfaceHandles {
    /// Append one Python view per leaf to `views` and one notify bit per leaf
    /// to `produced` (views order = legacy input order).
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()>;
}

impl<const N: usize> PyArgs for ArrayPort<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        views.push(NativeArrayView::bind_view::<N>(py, refs)?);
        produced.push(is_eventful(refs));
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPort<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        // A series is a behavior (always current), so its bit is always set.
        views.push(NativeSeriesView::bind::<N>(py, refs)?);
        produced.push(true);
        Ok(())
    }
}

impl PyArgs for ClockPort {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        views.push(py.None().into_bound(py));
        produced.push(refs);
        Ok(())
    }
}

/// A runtime-length group appends one view + bit per element.
impl<const N: usize> PyArgs for ArrayPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        for &value in refs {
            <ArrayPort<f64, N> as PyArgs>::append_views(py, value, views, produced)?;
        }
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        for &value in refs {
            <SeriesPort<f64, N> as PyArgs>::append_views(py, value, views, produced)?;
        }
        Ok(())
    }
}

macro_rules! tuple_pyargs {
    ($($idx:tt: $T:ident),+) => {
        impl<$($T: PyArgs,)+> PyArgs for ($($T,)+) {
            fn append_views<'py>(
                py: Python<'py>,
                refs: Self::Values<'_>,
                views: &mut Vec<Bound<'py, PyAny>>,
                produced: &mut Vec<bool>,
            ) -> PyResult<()> {
                $( $T::append_views(py, refs.$idx, views, produced)?; )+
                Ok(())
            }
        }
    };
}

tuple_pyargs!(0: A);
tuple_pyargs!(0: A, 1: B);
tuple_pyargs!(0: A, 1: B, 2: C);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
tuple_pyargs!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);
