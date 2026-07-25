//! [`PyArgs`] — build the Python view tuple + produced bools from typed input
//! refs.

use pyo3::prelude::*;

use crate::graph::typed::{Interface, InterfaceHandles};
use crate::ports::{ArrayPort, ArrayPorts, SeriesPort, SeriesPorts, UnitPort};

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
        let (notify, value) = refs;
        views.push(NativeArrayView::bind_view::<N>(py, value)?);
        produced.push(notify);
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
        let (notify, value) = refs;
        views.push(NativeSeriesView::bind::<N>(py, value)?);
        produced.push(notify);
        Ok(())
    }
}

impl PyArgs for UnitPort {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (notify, _) = refs;
        views.push(py.None().into_bound(py));
        produced.push(notify);
        Ok(())
    }
}

/// A runtime-length group appends one view + bit per element. (Concrete per
/// leaf type — a generic `ViewPorts<V> where ViewPort<V>: PyArgs` impl cannot
/// pass a `(bool, View)` tuple where the unnormalized
/// `<ViewPort<V> as Interface>::Values` projection is expected.)
impl<const N: usize> PyArgs for ArrayPorts<f64, N> {
    fn append_views<'py>(
        py: Python<'py>,
        refs: Self::Values<'_>,
        views: &mut Vec<Bound<'py, PyAny>>,
        produced: &mut Vec<bool>,
    ) -> PyResult<()> {
        let (flags, values) = refs;
        debug_assert!(
            flags.len() == values.len(),
            "ArrayPorts payload planes disagree on length"
        );
        for (i, &value) in values.iter().enumerate() {
            <ArrayPort<f64, N> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
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
        let (flags, values) = refs;
        debug_assert!(
            flags.len() == values.len(),
            "SeriesPorts refs planes disagree on length"
        );
        for (i, &value) in values.iter().enumerate() {
            <SeriesPort<f64, N> as PyArgs>::append_views(py, (flags[i], value), views, produced)?;
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
