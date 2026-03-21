//! Benchmark functions exposed to Python — mirrors [`bench_add.rs`](../../../benches/bench_add.rs).
//!
//! Each `#[pyfunction]` corresponds 1-to-1 to a criterion benchmark in
//! `benches/bench_add.rs`, running the same Rust code but callable from
//! Python.  Results are returned as [`BenchResult`] for validation.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::operators;
use crate::scenario::Scenario;
use crate::store::Store;
use crate::Operator;

type PyObject = Py<PyAny>;

// ---------------------------------------------------------------------------
// Result wrapper
// ---------------------------------------------------------------------------

/// Owns a copy of the computed values so Python can inspect them.
#[pyclass]
pub struct BenchResult {
    values: Vec<f64>,
}

#[pymethods]
impl BenchResult {
    /// Return values as a 1-D numpy float64 array.
    fn values_array<'py>(&self, py: Python<'py>) -> PyObject {
        let arr = Array1::from(self.values.clone());
        PyArray1::from_owned_array(py, arr).into_any().unbind()
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }
}

// ---------------------------------------------------------------------------
// Baseline: plain add (element-only)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_baseline_add(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    _ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let n = a.len();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a[i] + b[i]);
    }

    BenchResult { values: out }
}

// ---------------------------------------------------------------------------
// Baseline: plain add (series)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_baseline_add_series(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    _ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let n = a.len();

    let mut vec_a = Vec::new();
    let mut vec_b = Vec::new();
    let mut vec_out = Vec::new();
    for i in 0..n {
        vec_a.push(a[i]);
        vec_b.push(b[i]);
        vec_out.push(vec_a.last().unwrap() + vec_b.last().unwrap());
    }

    BenchResult { values: vec_out }
}

// ---------------------------------------------------------------------------
// Store with plain add (element-only)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_store_add(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut store_a = Store::element(&[], &[0.0_f64]);
    let mut store_b = Store::element(&[], &[0.0_f64]);
    let mut store_out = Store::element(&[], &[0.0_f64]);
    for i in 0..n {
        store_a.push(ts[i], &[a[i]]);
        store_b.push(ts[i], &[b[i]]);
        store_out.push(ts[i], &[store_a.current()[0] + store_b.current()[0]]);
    }

    BenchResult {
        values: store_out.current().to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Store with plain add (series)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_store_add_series(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut store_a = Store::series(&[], &[0.0_f64]);
    let mut store_b = Store::series(&[], &[0.0_f64]);
    let mut store_out = Store::series(&[], &[0.0_f64]);
    for i in 0..n {
        store_a.push(ts[i], &[a[i]]);
        store_b.push(ts[i], &[b[i]]);
        store_out.push(ts[i], &[store_a.current()[0] + store_b.current()[0]]);
    }

    BenchResult {
        values: store_out.values()[1..].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Store compute (element-only)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_store_compute(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut store_a = Store::element(&[], &[0.0_f64]);
    let mut store_b = Store::element(&[], &[0.0_f64]);
    let mut store_out = Store::element(&[], &[0.0_f64]);
    let mut state = operators::add::<f64>().init();
    for i in 0..n {
        store_a.push(ts[i], &[a[i]]);
        store_b.push(ts[i], &[b[i]]);
        store_out.push_default(ts[i]);
        let produced = <operators::Add<f64> as Operator>::compute(
            &mut state,
            (&store_a, &store_b),
            store_out.current_view_mut(),
        );
        if produced {
            store_out.commit();
        } else {
            store_out.rollback();
        }
    }

    BenchResult {
        values: store_out.current().to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Store compute (series)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_store_compute_series(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut store_a = Store::series(&[], &[0.0_f64]);
    let mut store_b = Store::series(&[], &[0.0_f64]);
    let mut store_out = Store::series(&[], &[0.0_f64]);
    let mut state = operators::add::<f64>().init();
    for i in 0..n {
        store_a.push(ts[i], &[a[i]]);
        store_b.push(ts[i], &[b[i]]);
        store_out.push_default(ts[i]);
        let produced = <operators::Add<f64> as Operator>::compute(
            &mut state,
            (&store_a, &store_b),
            store_out.current_view_mut(),
        );
        if produced {
            store_out.commit();
        } else {
            store_out.rollback();
        }
    }

    BenchResult {
        values: store_out.values()[1..].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Scenario operator (element-only)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_scenario_operator(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.create_node::<f64>(&[], &[0.0]);
    let hb = sc.create_node::<f64>(&[], &[0.0]);
    let ho = sc.add_operator([ha, hb], operators::add());
    for i in 0..n {
        sc.store_mut(ha).push(ts[i], &[a[i]]);
        sc.store_mut(hb).push(ts[i], &[b[i]]);
        sc.flush(ts[i], &[ha.index(), hb.index()]);
    }

    BenchResult {
        values: sc.store(ho).current().to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Scenario operator (series)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_scenario_operator_series(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.create_node::<f64>(&[], &[0.0]);
    let hb = sc.create_node::<f64>(&[], &[0.0]);
    let ho = sc.add_operator([ha, hb], operators::add());
    sc.store_mut(ho).ensure_min_window(0);
    for i in 0..n {
        sc.store_mut(ha).push(ts[i], &[a[i]]);
        sc.store_mut(hb).push(ts[i], &[b[i]]);
        sc.flush(ts[i], &[ha.index(), hb.index()]);
    }

    BenchResult {
        values: sc.store(ho).values()[1..].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Scenario chain (depth operators)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_scenario_chain(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
    depth: usize,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.create_node::<f64>(&[], &[0.0]);
    let hb = sc.create_node::<f64>(&[], &[0.0]);

    let mut prev = sc.add_operator([ha, hb], operators::add());
    for i in 1..depth {
        let other = if i % 2 == 0 { ha } else { hb };
        prev = sc.add_operator([prev, other], operators::add());
    }

    for i in 0..n {
        sc.store_mut(ha).push(ts[i], &[a[i]]);
        sc.store_mut(hb).push(ts[i], &[b[i]]);
        sc.flush(ts[i], &[ha.index(), hb.index()]);
    }

    BenchResult {
        values: sc.store(prev).current().to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Scenario sparse graph (many operators, few active)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn bench_scenario_sparse(
    a: PyReadonlyArray1<'_, f64>,
    b: PyReadonlyArray1<'_, f64>,
    ts: PyReadonlyArray1<'_, i64>,
    total: usize,
    active: usize,
) -> BenchResult {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    let ts = ts.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.create_node::<f64>(&[], &[0.0]);
    let hb = sc.create_node::<f64>(&[], &[0.0]);
    let hc = sc.create_node::<f64>(&[], &[0.0]);
    let hd = sc.create_node::<f64>(&[], &[0.0]);

    // Active chain
    let mut last = sc.add_operator([ha, hb], operators::add());
    for _ in 1..active {
        last = sc.add_operator([last, ha], operators::add());
    }

    // Inactive chain
    let inactive = total - active;
    if inactive > 0 {
        let mut prev = sc.add_operator([hc, hd], operators::add());
        for _ in 1..inactive {
            prev = sc.add_operator([prev, hc], operators::add());
        }
    }

    for i in 0..n {
        sc.store_mut(ha).push(ts[i], &[a[i]]);
        sc.store_mut(hb).push(ts[i], &[b[i]]);
        sc.flush(ts[i], &[ha.index(), hb.index()]);
    }

    BenchResult {
        values: sc.store(last).current().to_vec(),
    }
}
