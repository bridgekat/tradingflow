//! [`ScenarioExt`] — TradingFlow's domain sugar over the generic engine.
//!
//! The engine itself (graph builder + self-driving event-loop graph) is
//! `flowgraph::ingest`'s [`Builder`] / [`Graph`](flowgraph::ingest::Graph);
//! TradingFlow's [`Scenario`](crate::Scenario) / [`Session`](crate::Session)
//! are their TAI-[`Instant`] + [`WallClock`](crate::WallClock) instantiations.
//! This trait adds what is TradingFlow-specific: the
//! [`ArrayView`](crate::ArrayView) currency bridges, the
//! [`add_const`](ScenarioExt::add_const) cell sugar, and the Python operator
//! registrars. (Records register through the plain deref'd
//! [`push`](flowgraph::typed::Builder::push) with the
//! [`record`](crate::operators::record) /
//! [`record_bounded`](crate::operators::record_bounded) constructors, which
//! take the driver's [`Clock`](crate::operators::Clock) as a parameter.)

use flowgraph::ingest::{Builder, Clock as IngestClock};
use flowgraph::typed::{Handle, RefPort, RefSource, RefViewPort, SourceHandle, ViewPort};

use crate::operators::ArrayValue;
use crate::{Array, Instant, Scalar};

/// Extension methods for [`Builder<Instant, C>`] — i.e.
/// [`Scenario`](crate::Scenario): the Array view-currency bridges and the
/// TradingFlow operator registrars. Generic segments register via the deref'd
/// [`push`](flowgraph::typed::Builder::push); sources via
/// [`Builder::add_source`].
pub trait ScenarioExt {
    /// Register a constant node: an externally-set source cell with no feed
    /// (sugar for `push_source(RefSource::new(value))`). Mutate it via the
    /// session's deref'd [`state_mut`](flowgraph::typed::Graph::state_mut) +
    /// [`stabilize_at`](flowgraph::ingest::Graph::stabilize_at); wire it
    /// downstream via the deref'd plain handle (`*h`).
    fn add_const<V: Send + Sync + 'static>(&mut self, value: V) -> SourceHandle<RefSource<V>>;

    /// Bridge a whole-array reference handle (a source cell /
    /// [`add_const`](Self::add_const) / any `RefPort<Array<T, N>>`) into the
    /// `ViewPort<ArrayValue<T, N>>` view currency the operators speak, via a
    /// zero-copy [`AsView`](crate::operators::AsView). Source outputs are
    /// whole-array references; everything downstream is views.
    fn as_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefPort<Array<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>>;

    /// Bridge a by-value view handle into the by-reference `RefViewPort` currency
    /// the carry-join combines ([`Stack`](crate::operators::Stack) /
    /// [`Concat`](crate::operators::Concat)) consume, via
    /// [`RefArrayView`](crate::operators::RefArrayView). Lets independently-produced
    /// view handles (e.g. separate feature columns) be collected into the
    /// `&[RefViewPort]` slice those combines take.
    fn ref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefViewPort<ArrayValue<T, N>>>;

    /// Bridge a whole slice of by-value view handles into the `RefViewPort`
    /// handles a carry-join consumes — the vector form of
    /// [`ref_array_view`](Self::ref_array_view), for the common
    /// `sc.push(Stack::new(axis), &sc.ref_array_views(&columns)[..])` shape.
    fn ref_array_views<T: Scalar, const N: usize>(
        &mut self,
        data: &[Handle<ViewPort<ArrayValue<T, N>>>],
    ) -> Vec<Handle<RefViewPort<ArrayValue<T, N>>>> {
        data.iter().map(|&h| self.ref_array_view::<T, N>(h)).collect()
    }

    /// Re-derive a by-reference view handle (e.g. a
    /// [`Split`](crate::operators::Split) row) as the by-value `ViewPort` currency
    /// the elementwise operators consume, via
    /// [`DerefArrayView`](crate::operators::DerefArrayView). The inverse of
    /// [`ref_array_view`](Self::ref_array_view).
    fn deref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefViewPort<ArrayValue<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>>;

    /// Materialize a by-value view handle back into an **owned** whole-array
    /// `RefPort<Array<T, N>>` cell, via [`Own`](crate::operators::Own) — the
    /// inverse of [`as_view`](Self::as_view), e.g. before a whole-array consumer
    /// such as a `PyClassOperator`.
    fn own<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefPort<Array<T, N>>>;

    /// Register a Python operator in **return mode** (feature `python`).
    /// `source` is a Python expression evaluating to a callable that takes
    /// `inputs.len()` `float64` ndarrays and returns a 1-D `float64` ndarray of
    /// length `out_len`, e.g. `"lambda a, b: a + b"`. Operators run on the shared
    /// (free-threaded) interpreter and execute in parallel on the pool. See the
    /// `operators::python` module docs for the data model, retention contract, and setup.
    #[cfg(feature = "python")]
    fn add_py_operator(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>>;

    /// Register a Python operator in **write mode** (feature `python`, zero-copy
    /// output). `source` evaluates to a callable `f(out, *inputs)` taking a
    /// writable `float64` ndarray `out` (aliasing the output buffer) followed by
    /// the input ndarrays; it writes its `out_len` results into `out` in place
    /// and its return value is ignored, e.g. `"lambda out, a, b: np.add(a, b, out=out)"`.
    /// The `out` view must not escape the call (see the `operators::python` module docs).
    #[cfg(feature = "python")]
    fn add_py_operator_writing(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>>;

    /// Register a class-based Python operator (feature `python`). `source` is a
    /// Python program binding the operator instance to `__op__`, implementing the
    /// legacy contract `init(inputs, timestamp) -> state` and
    /// `compute(state, inputs, output, timestamp, produced) -> bool`. The driver
    /// [`Clock`](crate::operators::Clock) is wired through so the operator sees
    /// event time. `out_shape` is the output element shape (`[]` for a scalar).
    /// See [`PyClassOperator`](crate::operators::PyClassOperator).
    #[cfg(feature = "python")]
    fn add_py_class_operator(
        &mut self,
        source: &str,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>>;

    /// Register a class-based Python operator loaded from a plain `.py` file
    /// (feature `python`). The file defines `build(**kwargs)` (called with
    /// `params`) or binds `__op__`; see [`PyClassOperator`](crate::operators::PyClassOperator).
    /// Convenience for all-`Array<f64, 1>` inputs; heterogeneous (Array + Series)
    /// operators register via the deref'd
    /// [`push`](flowgraph::typed::Builder::push) with
    /// `PyClassOperator::<I, NO>::from_file(..)`.
    #[cfg(feature = "python")]
    fn add_py_operator_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>>;
}

impl<C: IngestClock<Instant>> ScenarioExt for Builder<Instant, C> {
    fn add_const<V: Send + Sync + 'static>(&mut self, value: V) -> SourceHandle<RefSource<V>> {
        self.push_source(RefSource::new(value))
    }

    fn as_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefPort<Array<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>> {
        self.push(crate::operators::AsView::<T, N>::new(), data)
    }

    fn ref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefViewPort<ArrayValue<T, N>>> {
        self.push(crate::operators::RefArrayView::<T, N>::new(), data)
    }

    fn deref_array_view<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<RefViewPort<ArrayValue<T, N>>>,
    ) -> Handle<ViewPort<ArrayValue<T, N>>> {
        self.push(crate::operators::DerefArrayView::<T, N>::new(), data)
    }

    fn own<T: Scalar, const N: usize>(
        &mut self,
        data: Handle<ViewPort<ArrayValue<T, N>>>,
    ) -> Handle<RefPort<Array<T, N>>> {
        self.push(crate::operators::Own::<T, N>::new(), data)
    }

    #[cfg(feature = "python")]
    fn add_py_operator(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.push(crate::operators::PyOperator::<1, 1>::new(source, out_len), inputs)
    }

    #[cfg(feature = "python")]
    fn add_py_operator_writing(
        &mut self,
        source: &str,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_len: usize,
    ) -> Handle<RefPort<Array<f64, 1>>> {
        self.push(
            crate::operators::PyOperator::<1, 1>::writing(source, out_len),
            inputs,
        )
    }

    #[cfg(feature = "python")]
    fn add_py_class_operator(
        &mut self,
        source: &str,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>> {
        let clock = self.time();
        self.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64, 1>>, 1>::from_source(
                source,
                params,
                out_shape.to_vec(),
                clock,
            ),
            inputs,
        )
    }

    #[cfg(feature = "python")]
    fn add_py_operator_file(
        &mut self,
        path: impl AsRef<std::path::Path>,
        params: crate::operators::PyParams,
        inputs: &[Handle<RefPort<Array<f64, 1>>>],
        out_shape: &[usize],
    ) -> Handle<RefPort<Array<f64, 1>>> {
        let clock = self.time();
        self.push(
            crate::operators::PyClassOperator::<flowgraph::typed::RefPorts<Array<f64, 1>>, 1>::from_file(
                path,
                params,
                out_shape.to_vec(),
                clock,
            ),
            inputs,
        )
    }
}
