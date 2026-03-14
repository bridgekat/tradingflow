//! Benchmark entry points exposed to Python via PyO3.
//!
//! These functions are called from `benches/bench_add.py` and exist purely
//! for performance measurement.  They are not part of the public API.

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::operators;
use crate::scenario::Scenario;
use crate::series::{Series, SeriesHandle};

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
    use crate::operator::Operator;

    let _ = py;
    let a = raw_a.as_slice().unwrap();
    let b = raw_b.as_slice().unwrap();
    let ts = timestamps_ns.as_slice().unwrap();
    let n = a.len();

    let mut sa = Series::with_capacity(&[], n);
    let mut sb = Series::with_capacity(&[], n);
    let mut op = operators::add();
    let mut out = Series::with_capacity(&[], n);
    let mut buf = [0.0f64; 1];

    for i in 0..n {
        sa.append_unchecked(ts[i], &[a[i]]);
        sb.append_unchecked(ts[i], &[b[i]]);
        if op.compute(ts[i], &[&sa, &sb], &mut buf) {
            out.append_unchecked(ts[i], &buf);
        }
    }

    NativeSeries {
        timestamps: out.timestamps_to_vec(),
        values: out.values_to_vec(),
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
    let ha: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    let hb: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    let ho: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    sc.add_apply(&[ha, hb], ho, operators::add());

    for i in 0..n {
        unsafe {
            sc.series_mut(ha).append_unchecked(ts[i], &[a[i]]);
            sc.series_mut(hb).append_unchecked(ts[i], &[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(ho) };
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
    let ha: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    let hb: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);

    let mut prev: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    sc.add_apply(&[ha, hb], prev, operators::add());
    for i in 1..depth {
        let next: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
        let other = if i % 2 == 0 { ha } else { hb };
        sc.add_apply(&[prev, other], next, operators::add());
        prev = next;
    }

    for i in 0..n {
        unsafe {
            sc.series_mut(ha).append_unchecked(ts[i], &[a[i]]);
            sc.series_mut(hb).append_unchecked(ts[i], &[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(prev) };
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
    let ha: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    let hb: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    let hc: SeriesHandle<f64> = sc.add_series_with_capacity(&[], 0);
    let hd: SeriesHandle<f64> = sc.add_series_with_capacity(&[], 0);

    // Active chain
    let mut last_active: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
    sc.add_apply(&[ha, hb], last_active, operators::add());
    for _ in 1..active_ops {
        let next: SeriesHandle<f64> = sc.add_series_with_capacity(&[], n);
        sc.add_apply(&[last_active, ha], next, operators::add());
        last_active = next;
    }

    // Inactive chain (never triggered)
    let inactive_count = total_ops.saturating_sub(active_ops);
    if inactive_count > 0 {
        let mut prev: SeriesHandle<f64> = sc.add_series_with_capacity(&[], 0);
        sc.add_apply(&[hc, hd], prev, operators::add());
        for _ in 1..inactive_count {
            let next: SeriesHandle<f64> = sc.add_series_with_capacity(&[], 0);
            sc.add_apply(&[prev, hc], next, operators::add());
            prev = next;
        }
    }

    for i in 0..n {
        unsafe {
            sc.series_mut(ha).append_unchecked(ts[i], &[a[i]]);
            sc.series_mut(hb).append_unchecked(ts[i], &[b[i]]);
        }
        sc.flush(ts[i], &[ha.index, hb.index]);
    }

    let out = unsafe { sc.series_ref(last_active) };
    NativeSeries {
        timestamps: out.timestamps_to_vec(),
        values: out.values_to_vec(),
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
    Ok(())
}
