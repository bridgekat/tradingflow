//! Cross-sectional feature panels: the canonical 7-factor set and the CICC
//! handbook catalogs, each stacked into `(N, F)` and recorded on the daily pulse.

use tradingflow::graph::{Handle, RefPort};

use tradingflow::data::Duration;
use tradingflow::operators::{
    SeriesPort, Window, divide, lag_series, log, ma, ma_time, multiply, percentile, record_bounded,
    resample_view, rolling_variance, sqrt, stack, subtract, winsorize,
};
use tradingflow::{Retention, Scenario, Series};

use super::data::Stacked;
use super::{AvH, RETAIN_MARGIN, factors, pv_factors};

/// A cross-sectional feature panel.
pub struct Features {
    /// Per-feature live view handles (each `(num_stocks,)`), in column order.
    pub names: Vec<String>,
    pub handles: Vec<AvH>,
    /// Trading-day-aligned `Record` of the `(num_stocks, n_features)` panel.
    pub series: Handle<SeriesPort<f64, 2>>,
}

/// Stack feature columns into `(N, F)`, resample onto the daily close pulse, and
/// record under `retention` — the shared tail of every feature-panel builder.
fn stack_and_record(
    sc: &mut Scenario,
    st: &Stacked,
    names: Vec<String>,
    handles: Vec<AvH>,
    retention: Retention,
) -> Features {
    let stacked = sc.push(stack(1), &handles[..]);
    let sampled = sc.push(resample_view(), (st.adjusted_close, stacked));
    let series = sc.push(record_bounded(retention), sampled);
    Features {
        names,
        handles,
        series,
    }
}

/// Build the canonical factor panel (3 percentile-ranked fundamentals plus
/// momentum / volatility / turnover-MA / volume-ratio), recorded on the daily
/// trading pulse. The returned panel `Series` is recorded under
/// `feature_retention` (pass [`Retention::UNBOUNDED`] when a consumer needs full
/// history); the internal rolling-input records are bounded to their own windows.
pub fn build_features(
    sc: &mut Scenario,
    st: &Stacked,
    window: usize,
    feature_retention: Retention,
) -> Features {
    let win_ret = Retention::count(window + RETAIN_MARGIN);

    // Fundamentals.
    let market_cap = sc.push(multiply(), (st.close, st.total_shares));
    let bp = sc.push(divide(), (st.parent_equity, market_cap));
    // TTM net profit: a self-recording 365-day rolling mean.
    let net_profit_ttm = sc.push(ma_time(Duration::from_days(365)), st.net_profit);
    let ttm_roe = sc.push(divide(), (net_profit_ttm, st.parent_equity));

    // Momentum: window-period log return of adjusted close.
    let log_adj = sc.push(log(), st.adjusted_close);
    let log_adj_series = sc.push(record_bounded(win_ret), log_adj);
    let log_adj_lag = sc.push(lag_series(window, f64::NAN), log_adj_series);
    let momentum = sc.push(subtract(), (log_adj, log_adj_lag));

    // Volatility: rolling std of daily log returns.
    let log_var = sc.push(rolling_variance(Window::Count(window)), log_adj_series);
    let volatility = sc.push(sqrt(), log_var);

    // Turnover MA (self-recording).
    let turnover = sc.push(divide(), (st.volume, st.circ_shares));
    let turnover_ma = sc.push(ma(window), turnover);

    // Volume ratio (self-recording MA).
    let volume_ma = sc.push(ma(window), st.volume);
    let volume_ratio = sc.push(divide(), (st.volume, volume_ma));

    let p = 0.01;
    let names = vec![
        "rank_market_cap".to_string(),
        "rank_bp".to_string(),
        "rank_ttm_roe".to_string(),
        format!("momentum_ma_{window}"),
        format!("volatility_{window}"),
        format!("turnover_ma_{window}"),
        format!("volume_ratio_{window}"),
    ];
    let handles = vec![
        sc.push(percentile(), market_cap),
        sc.push(percentile(), bp),
        sc.push(percentile(), ttm_roe),
        sc.push(winsorize(p), momentum),
        sc.push(winsorize(p), volatility),
        sc.push(winsorize(p), turnover_ma),
        sc.push(winsorize(p), volume_ratio),
    ];

    stack_and_record(sc, st, names, handles, feature_retention)
}

/// Which factor panel a strategy regresses on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FeatureSet {
    /// The canonical 7-factor [`build_features`] panel (back-compat default).
    Canonical,
    /// A curated, style-diverse ~24-factor subset of the CICC handbooks
    /// (value / quality / growth / size / momentum / reversal / low-vol /
    /// liquidity / illiquidity / price-volume-correlation / chip), chosen from
    /// the single-factor study. Within-style collinearity is handled by the
    /// predictor's pool-standardized Ridge (`alpha`).
    Cicc,
    /// All 145 CICC factors (fundamental + price-volume). Heavily collinear by
    /// design — leans entirely on Ridge to regularise.
    All,
}

impl FeatureSet {
    pub fn parse(s: &str) -> FeatureSet {
        match s {
            "canonical" => FeatureSet::Canonical,
            "cicc" => FeatureSet::Cicc,
            "all" => FeatureSet::All,
            other => panic!("unknown --feature-set {other:?} (expected canonical|cicc|all)"),
        }
    }
}

/// Curated style-diverse subset of CICC handbook factors used by
/// [`FeatureSet::Cicc`]. Names must match the catalog entries in
/// [`factors::build_factor_catalog`] / [`pv_factors::build_pv_catalog`].
const CICC_SUBSET: &[&str] = &[
    // Value / quality / growth / size (fundamental)
    "BP_LR",
    "SP_TTM",
    "EP_TTM",
    "ROE_TTM",
    "GPM_TTM",
    "CFOA",
    "APR_TTM",
    "TA_YOY",
    "Ln_MC",
    // Momentum / reversal (price-volume)
    "mmt_normal_M",
    "mmt_intraday_M",
    "mmt_overnight_M",
    "mmt_range_M",
    "mmt_route_M",
    // Volatility
    "vol_std_1M",
    "vol_up_std_1M",
    "vol_w_downshadow_std_1M",
    // Liquidity / illiquidity
    "liq_turn_std_1M",
    "liq_amihud_avg_1M",
    "liq_vstd_1M",
    // Price-volume correlation
    "corr_price_turn_1M",
    "corr_ret_turnd_1M",
    // Chip distribution
    "distribution_loss_l",
    "distribution_ret_avg",
];

/// Build a cross-sectionally percentile-ranked factor panel from the CICC
/// handbook catalogs, recorded on the daily close pulse — a drop-in [`Features`]
/// for the mean / mean-variance predictors. Each catalog entry is already the
/// model-ready feature (cross-sectionally ranked, missing values imputed to the
/// neutral median, so a stock with partial factor coverage stays predictable
/// rather than dropped by the predictor's all-features-finite mask). `set` must
/// be [`FeatureSet::Cicc`] or [`FeatureSet::All`] (use [`build_features`] for
/// [`FeatureSet::Canonical`]).
pub fn build_factor_features(
    sc: &mut Scenario,
    st: &Stacked,
    set: FeatureSet,
    feature_retention: Retention,
) -> Features {
    let fund = factors::build_factor_catalog(sc, st);
    let pv = pv_factors::build_pv_catalog(sc, st);
    let mut all_names = fund.names;
    let mut all_feat = fund.feature;
    all_names.extend(pv.names);
    all_feat.extend(pv.feature);

    let selected: Vec<usize> = match set {
        FeatureSet::All => (0..all_names.len()).collect(),
        FeatureSet::Cicc => CICC_SUBSET
            .iter()
            .map(|nm| {
                all_names
                    .iter()
                    .position(|x| x == nm)
                    .unwrap_or_else(|| panic!("CICC_SUBSET names an unknown factor {nm:?}"))
            })
            .collect(),
        FeatureSet::Canonical => {
            panic!("build_factor_features called with Canonical; use build_features")
        }
    };

    let mut names = Vec::with_capacity(selected.len());
    let mut handles = Vec::with_capacity(selected.len());
    for &i in &selected {
        names.push(all_names[i].clone());
        handles.push(all_feat[i]);
    }

    stack_and_record(sc, st, names, handles, feature_retention)
}

/// Build the strategy feature panel for `set`, dispatching to [`build_features`]
/// (canonical) or [`build_factor_features`] (CICC subsets). `window` is only
/// used by the canonical set.
pub fn build_strategy_features(
    sc: &mut Scenario,
    st: &Stacked,
    window: usize,
    set: FeatureSet,
    feature_retention: Retention,
) -> Features {
    match set {
        FeatureSet::Canonical => build_features(sc, st, window, feature_retention),
        _ => build_factor_features(sc, st, set, feature_retention),
    }
}
