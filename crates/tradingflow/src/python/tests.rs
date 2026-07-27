//! End-to-end tests: the host driven through the graph engine against the
//! embedded interpreter (spans the views, [`PyParams`] and [`PyClassOperator`]).

use super::{PyClassOperator, PyParams};
use crate::data::SeriesView;
use crate::data::{Array, Duration, Instant};
use crate::graph::pool::Pool;
use crate::graph::typed::Builder;
use crate::operators::{array, series};
use crate::ports::{ArrayPort, ArrayPorts, SeriesPort};

/// The last row of a recorded event stream — the python operators emit event
/// arrays, whose wires reset to `NaN` after every generation, so the value an
/// assertion wants is the newest row of a record behind the operator.
fn last_row<const N: usize>(v: SeriesView<'_, f64, N>) -> Vec<f64> {
    let rows = v.to_contiguous();
    let width = v.extents().iter().product::<usize>().max(1);
    assert!(!rows.is_empty(), "no event has been recorded yet");
    rows[rows.len() - width..].to_vec()
}

/// A small stateful operator over one Array input (L1 turnover), used here
/// purely as a from_source `PyClassOperator` fixture. Raw string at column 0
/// preserves Python indentation (Rust `\`-continuation would strip it).
const TURNOVER: &str = r#"
import numpy as np
from dataclasses import dataclass
@dataclass
class S:
    prev: object = None
    initialized: bool = False
class Turnover:
    def init(self, inputs, timestamp):
        return S()
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        current = np.where(np.isfinite(inputs[0].value()), inputs[0].value(), 0.0)
        if not state.initialized:
            state.prev = current
            state.initialized = True
            return False
        turnover = float(np.sum(np.abs(current - state.prev)))
        state.prev = current
        output.write(np.array(turnover, dtype=np.float64))
        return True
__op__ = Turnover()
"#;

#[test]
fn py_class_operator_turnover() {
    let mut b = Builder::new();
    let (src_cell, src) = b.source(array::from_parts([2], vec![0.5_f64, 0.5].into()));
    let out = b.segment(
        // Output is a scalar (`vec![]`), so NO = 0.
        PyClassOperator::<ArrayPorts<f64, 1>, 0>::from_source(TURNOVER, PyParams::new(), vec![]),
        &[src][..],
    );
    let rec = b.segment(series::record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src_cell) = Array::from_parts([2], vec![0.5, 0.5].into());
    g.stabilize(&mut pool, &Instant::MIN);
    // Warmup: the Python side returns `False`, i.e. produced no event, so
    // nothing is recorded at all (the old model published a 0.0 placeholder).
    assert_eq!(
        g.view(rec).to_contiguous().len(),
        0,
        "warmup emits no event"
    );

    *g.state_mut(src_cell) = Array::from_parts([2], vec![0.3, 0.7].into());
    g.stabilize(&mut pool, &Instant::MIN);
    assert!((&last_row(g.view(rec))[..][0] - 0.4).abs() < 1e-12);

    *g.state_mut(src_cell) = Array::from_parts([2], vec![1.0, 0.0].into());
    g.stabilize(&mut pool, &Instant::MIN);
    assert!((&last_row(g.view(rec))[..][0] - 1.4).abs() < 1e-12);
}

/// Heterogeneous inputs: an (Array, Series) operator that reads Series
/// history. Proves NativeSeriesView (values/len/getitem) + tuple PyArgs.
/// Computes: output = mean over history of (series[-1] dotted with weights).
const HIST_DOT: &str = r#"
import numpy as np
class HistDot:
    def init(self, inputs, timestamp):
        return {}
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        weights = inputs[0].value()          # (N,)
        hist = inputs[1].values()            # (T, N)
        # mean over time of <hist[t], weights>
        val = float(np.mean(hist @ weights)) if len(inputs[1]) > 0 else 0.0
        output.write(np.array(val, dtype=np.float64))
        return True
__op__ = HistDot()
"#;

#[test]
fn py_class_operator_heterogeneous_series() {
    let mut b = Builder::new();
    // weights: Array(2); feed_data: Array(2) recorded into a Series(2).
    // Sources lend the view currency directly — `Record` wires straight on.
    let (weights_cell, weights) = b.source(array::from_parts([2], vec![1.0_f64, 1.0].into()));
    let (feed_cell, feed) = b.source(array::from_parts([2], vec![0.0_f64, 0.0].into()));
    let series = b.segment(series::record_all(), feed);
    let out = b.segment(
        // Scalar output (`vec![]`), so NO = 0.
        PyClassOperator::<(ArrayPort<f64, 1>, SeriesPort<f64, 1>), 0>::from_source(
            HIST_DOT,
            PyParams::new(),
            vec![],
        ),
        (weights, series),
    );
    let rec = b.segment(series::record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Tick 1 @ t=100: feed [1,2]; series=[[1,2]]; dot with [1,1]=3; mean=3.
    let ctx = Instant::from_offset(Duration::from_nanos(100));
    *g.state_mut(weights_cell) = Array::from_parts([2], vec![1.0, 1.0].into());
    *g.state_mut(feed_cell) = Array::from_parts([2], vec![1.0, 2.0].into());
    g.stabilize(&mut pool, &ctx);
    assert!((&last_row(g.view(rec))[..][0] - 3.0).abs() < 1e-12);

    // Tick 2 @ t=200: feed [3,4]; series=[[1,2],[3,4]]; dots=3,7; mean=5.
    let ctx = Instant::from_offset(Duration::from_nanos(200));
    *g.state_mut(feed_cell) = Array::from_parts([2], vec![3.0, 4.0].into());
    g.stabilize(&mut pool, &ctx);
    assert!((&last_row(g.view(rec))[..][0] - 5.0).abs() < 1e-12);
}

/// Loading an operator from a plain `.py` file via a `build(**kwargs)`
/// factory parameterized from Rust with [`PyParams`].
#[test]
fn py_class_operator_from_file_with_params() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scaler.py");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(
            br#"
import numpy as np
class Scaler:
    def __init__(self, scale):
        self.scale = scale
    def init(self, inputs, timestamp):
        return {"scale": self.scale}
    @staticmethod
    def compute(state, inputs, output, timestamp, produced):
        total = float(np.sum(inputs[0].value())) * state["scale"]
        output.write(np.array(total, dtype=np.float64))
        return True
def build(scale=1.0):
    return Scaler(scale)
"#,
        )
        .unwrap();

    let mut b = Builder::new();
    let (src_cell, src) = b.source(array::from_parts([4], vec![1.0_f64, 2.0, 3.0, 4.0].into()));
    let out = b.segment(
        // Scalar output (`vec![]`), so NO = 0.
        PyClassOperator::<ArrayPorts<f64, 1>, 0>::from_file(
            &path,
            PyParams::new().float("scale", 3.0),
            vec![],
        ),
        &[src][..],
    );
    let rec = b.segment(series::record_all(), out);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    *g.state_mut(src_cell) = Array::from_parts([4], vec![1.0, 2.0, 3.0, 4.0].into());
    g.stabilize(&mut pool, &Instant::MIN);
    // sum(1..4)=10 * 3
    assert!((&last_row(g.view(rec))[..][0] - 30.0).abs() < 1e-12);
}

// -- Integration with the real `tradingflow` Python package (needs python/ --
//    on PYTHONPATH + numpy on the embedded interpreter's path). --------------

/// A real predictor (LinearRegression) end-to-end: Array universe + two
/// Series-history inputs (features (N,F), target (N)), produced-gated
/// rebalance, emitting an (N,) prediction. Validates the heterogeneous
/// Series path through the engine.
#[test]
fn pyhost_linear_regression_predictor() {
    const N: usize = 3;
    const F: usize = 2;
    let mut b = Builder::new();
    let (universe_cell, universe) = b.source(array::from_parts([N], vec![1.0; N].into()));
    let (feat_feed_cell, feat_feed) = b.source(array::zeros::<f64, 2>([N, F]));
    let (tgt_feed_cell, tgt_feed) = b.source(array::zeros::<f64, 1>([N]));
    // Sources lend the view currency directly — `Record` wires straight on.
    let feat_series = b.segment(series::record_all(), feat_feed);
    let tgt_series = b.segment(series::record_all(), tgt_feed);
    let pred = b.segment(
        // Output is the (N,) prediction → NO = 1 (the default).
        PyClassOperator::<(ArrayPort<f64, 1>, SeriesPort<f64, 2>, SeriesPort<f64, 1>)>::from_module(
            "tradingflow.predictors.mean.linear_regression",
            PyParams::new()
                .int("num_stocks", N as i64)
                .int("num_features", F as i64)
                .int("universe_size", N as i64)
                .int("target_offset", 1),
            vec![N],
        ),
        (universe, feat_series, tgt_series),
    );
    let rec = b.segment(series::record_all(), pred);
    let mut g = b.build();
    let mut pool = Pool::new(0);

    // Feed a few ticks of features/targets with a linear relationship so the
    // pooled OLS is well-posed; rebalance each tick (universe produces).
    for t in 1..=5_i64 {
        let x: Vec<f64> = (0..N * F).map(|k| (t as f64) + 0.1 * k as f64).collect();
        let y: Vec<f64> = (0..N).map(|i| 0.5 * (t as f64) + i as f64).collect();
        let ctx = Instant::from_offset(Duration::from_nanos(t * 100));
        *g.state_mut(feat_feed_cell) = Array::from_parts([N, F], x.into());
        *g.state_mut(tgt_feed_cell) = Array::from_parts([N], y.into());
        *g.state_mut(universe_cell) = Array::from_parts([N], vec![1.0; N].into());
        g.stabilize(&mut pool, &ctx);
    }

    let mu = last_row(g.view(rec));
    assert_eq!(mu.len(), N);
    assert!(
        mu.iter().all(|v| v.is_finite()),
        "prediction has non-finite entries: {mu:?}"
    );
}
