//! Benchmark entry points exposed to Python via PyO3.
//!
//! These functions are called from `benches/bench_add.py` and exist purely
//! for performance measurement.  They are not part of the public API.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::operators;
use crate::scenario::Scenario;
use crate::series::Series;

// ---------------------------------------------------------------------------
// Python-visible result type
// ---------------------------------------------------------------------------

#[pyclass]
pub struct NativeSeries {
    timestamps: Vec<i64>,
    values: Vec<f64>,
}

#[pymethods]
impl NativeSeries {
    #[new]
    fn py_new() -> Self {
        Self {
            timestamps: Vec::new(),
            values: Vec::new(),
        }
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn timestamps_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_array(py, &Array1::from_vec(self.timestamps.clone()))
    }

    fn values_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_array(py, &Array1::from_vec(self.values.clone()))
    }
}

// ---------------------------------------------------------------------------
// Benchmark functions
// ---------------------------------------------------------------------------

/// Direct operator compute loop (no Scenario).
#[pyfunction]
pub fn bench_add_compute<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    use crate::observable::Observable;
    use crate::operator::Operator;

    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut obs_a = Observable::new(&[], &[0.0]);
    let mut obs_b = Observable::new(&[], &[0.0]);
    let mut op = operators::add();
    let mut out_series = Series::with_capacity(&[], n);
    let mut buf = [0.0f64; 1];

    for i in 0..n {
        obs_a.write(&[a[i]]);
        obs_b.write(&[b[i]]);
        if op.compute(ts[i], &[&obs_a, &obs_b], &mut buf) {
            out_series.append_unchecked(ts[i], &buf);
        }
    }

    NativeSeries {
        timestamps: out_series.timestamps_to_vec(),
        values: out_series.values_to_vec(),
    }
}

/// Scenario-based compute (single add operator).
#[pyfunction]
pub fn bench_scenario_compute<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);
    let ho = sc.add_apply(&[ha, hb], operators::add());
    let ho_series = sc.materialize_with_capacity::<f64>(ho, n);

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(ho_series) };
    NativeSeries {
        timestamps: out.timestamps_to_vec(),
        values: out.values_to_vec(),
    }
}

/// 3 series built in 3 separate loops (baseline: sequential writes).
#[pyfunction]
pub fn bench_add_loop<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sa_ts = Vec::with_capacity(n);
    let mut sa_vals = Vec::with_capacity(n);
    for i in 0..n {
        sa_ts.push(ts[i]);
        sa_vals.push(a[i]);
    }
    let mut sb_ts = Vec::with_capacity(n);
    let mut sb_vals = Vec::with_capacity(n);
    for i in 0..n {
        sb_ts.push(ts[i]);
        sb_vals.push(b[i]);
    }
    let mut out_ts = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    for i in 0..n {
        out_ts.push(ts[i]);
        out_vals.push(sa_vals[i] + sb_vals[i]);
    }

    let _ = (py, sa_ts, sb_ts, sb_vals);
    NativeSeries {
        timestamps: out_ts,
        values: out_vals,
    }
}

/// 3 series built simultaneously in 1 loop (matches compute/scenario access
/// pattern).
#[pyfunction]
pub fn bench_add_loop_interleaved<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sa_ts = Vec::with_capacity(n);
    let mut sa_vals = Vec::with_capacity(n);
    let mut sb_ts = Vec::with_capacity(n);
    let mut sb_vals = Vec::with_capacity(n);
    let mut out_ts = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    for i in 0..n {
        sa_ts.push(ts[i]);
        sa_vals.push(a[i]);
        sb_ts.push(ts[i]);
        sb_vals.push(b[i]);
        out_ts.push(ts[i]);
        out_vals.push(a[i] + b[i]);
    }

    let _ = (py, sa_ts, sa_vals, sb_ts, sb_vals);
    NativeSeries {
        timestamps: out_ts,
        values: out_vals,
    }
}

/// Opaque fn-ptr compute: mimics type-erased Scenario dispatch path.
unsafe fn add_erased(inputs: *const *const u8, output: *mut u8) {
    unsafe {
        let a = *(inputs.add(0) as *const *const f64);
        let b = *(inputs.add(1) as *const *const f64);
        let out = output as *mut f64;
        *out = *a + *b;
    }
}

/// 3 series interleaved + fn-ptr dispatch (closest to scenario without Series
/// struct).
#[pyfunction]
pub fn bench_add_loop_fnptr<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sa_ts = Vec::with_capacity(n);
    let mut sa_vals = Vec::with_capacity(n);
    let mut sb_ts = Vec::with_capacity(n);
    let mut sb_vals = Vec::with_capacity(n);
    let mut out_ts = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    let f: unsafe fn(*const *const u8, *mut u8) = add_erased;
    unsafe {
        for i in 0..n {
            sa_ts.push(ts[i]);
            sa_vals.push(a[i]);
            sb_ts.push(ts[i]);
            sb_vals.push(b[i]);
            let pa = sa_vals.as_ptr().add(i) as *const u8;
            let pb = sb_vals.as_ptr().add(i) as *const u8;
            let inputs = [pa, pb];
            let mut val: f64 = 0.0;
            f(inputs.as_ptr(), &mut val as *mut f64 as *mut u8);
            out_ts.push(ts[i]);
            out_vals.push(val);
        }
    }

    let _ = (py, sa_ts, sa_vals, sb_ts, sb_vals);
    NativeSeries {
        timestamps: out_ts,
        values: out_vals,
    }
}

/// Scenario with a chain of `depth` add operators.
#[pyfunction]
pub fn bench_scenario_chain<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    depth: usize,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);

    let mut prev = sc.add_apply(&[ha, hb], operators::add());
    for i in 1..depth {
        let other = if i % 2 == 0 { ha } else { hb };
        prev = sc.add_apply(&[prev, other], operators::add());
    }
    let prev_series = sc.materialize_with_capacity::<f64>(prev, n);

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(prev_series) };
    NativeSeries {
        timestamps: out.timestamps_to_vec(),
        values: out.values_to_vec(),
    }
}

/// Sparse graph: `total_ops` operators but only `active_ops` are downstream
/// of the sources that actually update.
#[pyfunction]
pub fn bench_scenario_sparse<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    total_ops: usize,
    active_ops: usize,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);
    let hc = sc.add_source::<f64>(&[], &[0.0]);
    let hd = sc.add_source::<f64>(&[], &[0.0]);

    // Active chain
    let mut last_active = sc.add_apply(&[ha, hb], operators::add());
    for _ in 1..active_ops {
        last_active = sc.add_apply(&[last_active, ha], operators::add());
    }
    let last_active_series = sc.materialize_with_capacity::<f64>(last_active, n);

    // Inactive chain (never triggered)
    let inactive_count = total_ops.saturating_sub(active_ops);
    if inactive_count > 0 {
        let mut prev = sc.add_apply(&[hc, hd], operators::add());
        for _ in 1..inactive_count {
            prev = sc.add_apply(&[prev, hc], operators::add());
        }
    }

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(last_active_series) };
    NativeSeries {
        timestamps: out.timestamps_to_vec(),
        values: out.values_to_vec(),
    }
}

// ---------------------------------------------------------------------------
// Observable-only variants (no materialization — measures overhead savings)
// ---------------------------------------------------------------------------

/// Direct operator compute loop, output stays as observable (no Series).
#[pyfunction]
pub fn bench_add_compute_obs<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    use crate::observable::Observable;
    use crate::operator::Operator;

    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut obs_a = Observable::new(&[], &[0.0]);
    let mut obs_b = Observable::new(&[], &[0.0]);
    let mut obs_out = Observable::new(&[], &[0.0]);
    let mut op = operators::add();

    for i in 0..n {
        obs_a.write(&[a[i]]);
        obs_b.write(&[b[i]]);
        op.compute(ts[i], &[&obs_a, &obs_b], obs_out.vals_mut());
    }

    // Return only the last value for verification.
    NativeSeries {
        timestamps: vec![ts[n - 1]],
        values: vec![obs_out.last()[0]],
    }
}

/// Scenario single add, output NOT materialized.
#[pyfunction]
pub fn bench_scenario_compute_obs<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);
    let ho = sc.add_apply(&[ha, hb], operators::add());
    // No materialize — output is observable only.

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.observable_ref(ho) };
    NativeSeries {
        timestamps: vec![ts[n - 1]],
        values: vec![out.last()[0]],
    }
}

/// Scenario chain of `depth` add operators, NO materialization anywhere.
#[pyfunction]
pub fn bench_scenario_chain_obs<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    depth: usize,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);

    let mut prev = sc.add_apply(&[ha, hb], operators::add());
    for i in 1..depth {
        let other = if i % 2 == 0 { ha } else { hb };
        prev = sc.add_apply(&[prev, other], operators::add());
    }
    // No materialize — all nodes are observable only.

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.observable_ref(prev) };
    NativeSeries {
        timestamps: vec![ts[n - 1]],
        values: vec![out.last()[0]],
    }
}

/// Sparse graph, NO materialization anywhere.
#[pyfunction]
pub fn bench_scenario_sparse_obs<'py>(
    py: Python<'py>,
    raw_a: PyReadonlyArray1<'py, f64>,
    raw_b: PyReadonlyArray1<'py, f64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    total_ops: usize,
    active_ops: usize,
) -> NativeSeries {
    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sc = Scenario::new();
    let ha = sc.add_source::<f64>(&[], &[0.0]);
    let hb = sc.add_source::<f64>(&[], &[0.0]);
    let hc = sc.add_source::<f64>(&[], &[0.0]);
    let hd = sc.add_source::<f64>(&[], &[0.0]);

    // Active chain
    let mut last_active = sc.add_apply(&[ha, hb], operators::add());
    for _ in 1..active_ops {
        last_active = sc.add_apply(&[last_active, ha], operators::add());
    }
    // No materialize.

    // Inactive chain (never triggered)
    let inactive_count = total_ops.saturating_sub(active_ops);
    if inactive_count > 0 {
        let mut prev = sc.add_apply(&[hc, hd], operators::add());
        for _ in 1..inactive_count {
            prev = sc.add_apply(&[prev, hc], operators::add());
        }
    }

    for i in 0..n {
        unsafe {
            sc.observable_mut(ha).write(&[a[i]]);
            sc.observable_mut(hb).write(&[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.observable_ref(last_active) };
    NativeSeries {
        timestamps: vec![ts[n - 1]],
        values: vec![out.last()[0]],
    }
}

/// Register all benchmark functions on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeSeries>()?;
    m.add_function(pyo3::wrap_pyfunction!(bench_add_compute, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_compute, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_add_loop, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_add_loop_interleaved, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_add_loop_fnptr, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_chain, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_sparse, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_add_compute_obs, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_compute_obs, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_chain_obs, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(bench_scenario_sparse_obs, m)?)?;
    Ok(())
}
