//! [`PyArgs`] — build the unified Python input tuple from typed input refs.

use pyo3::prelude::*;

use crate::graph::Interface;
use crate::graph::typed::InterfaceHandles;
use crate::ports::{ArrayPort, ArrayPorts, SeriesPort, SeriesPorts, SignalPort, SignalPorts};

use super::{NativeArrayView, NativeArrayViewBool, NativeSeriesView};

/// Walks an operator's input [`Interface`] refs tree, appending **one Python
/// value per leaf, in tree order**: a [`NativeArrayView`] /
/// [`NativeSeriesView`] for a data leaf, and a [`NativeArrayViewBool`] for a
/// signal leaf of any rank (a rank-0 signal reads as `if signal:` via its
/// `__bool__`; a signal array reads as a NumPy bool mask via `value()`). The
/// Python side receives a single `inputs` tuple whose positions mirror the
/// wiring exactly — signals and values share one numbering.
pub trait PyArgs: Interface + InterfaceHandles {
    /// Append this leaf's (or subtree's) Python values to `views` in tree
    /// order.
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()>;
}

impl<const N: usize> PyArgs for ArrayPort<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        views.push(NativeArrayView::bind_view::<N>(py, refs)?);
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPort<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        views.push(NativeSeriesView::bind::<N>(py, refs)?);
        Ok(())
    }
}

impl<const N: usize> PyArgs for SignalPort<N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        // The pulses ride the inputs tuple as a bool view, in wiring order.
        views.push(NativeArrayViewBool::bind_view::<N>(py, refs)?);
        Ok(())
    }
}

/// A runtime-length group appends one view per element.
impl<const N: usize> PyArgs for ArrayPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for &value in refs {
            <ArrayPort<f64, N> as PyArgs>::append_views(py, value, views)?;
        }
        Ok(())
    }
}

impl<const N: usize> PyArgs for SeriesPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for &value in refs {
            <SeriesPort<f64, N> as PyArgs>::append_views(py, value, views)?;
        }
        Ok(())
    }
}

impl<const N: usize> PyArgs for SignalPorts<N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
    ) -> PyResult<()> {
        for &value in refs {
            <SignalPort<N> as PyArgs>::append_views(py, value, views)?;
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
            ) -> PyResult<()> {
                $( $T::append_views(py, refs.$idx, views)?; )+
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
