use crate::data::{ArrayView, Instant};
use crate::graph::{Interface, Operator};
use crate::ports::{ArrayPort, SignalPort};

/// Base trait for fixed-price execution policies.
///
/// Orders are assumed to execute at exactly the given prices.
pub trait Exec: Send + 'static {
    /// Generates orders based on current prices.
    ///
    /// - Prices and cash are given in [`f64`] of some unit (positive).
    /// - Positions are given in [`f64`] of some unit (positive).
    /// - Target values are given in the same unit as prices and cash.
    ///
    /// No input is ever `NaN` (a `NaN` quote is a data error, and the engine
    /// panics on one before calling this). It is guaranteed that
    /// `bids[i] <= asks[i]` for all `i`. If `bids[i] == -inf` or
    /// `asks[i] == +inf`, then the `i`-th stock cannot be sold or bought,
    /// respectively — a policy may still emit such an order, but the engine
    /// drops it.
    ///
    /// Returns a list of `(i, delta)` where `delta` is the change in `i`-th
    /// position: positive for buy, negative for sell.
    fn orders(
        &mut self,
        bids: &[f64],
        asks: &[f64],
        positions: &[f64],
        cash: f64,
        target_values: &[f64],
    ) -> Vec<(usize, f64)>;
}

/// Operator parameters for fixed-price traders.
#[derive(Debug, Clone, Copy, Default)]
pub struct FixedParams {
    pub delayed: bool,
    pub initial_cash: f64,
    pub fee_base_buy: f64,
    pub fee_base_sell: f64,
    pub fee_rate_buy: f64,
    pub fee_rate_sell: f64,
}

/// Operator signature for fixed-price traders.
pub struct Fixed<E: Exec> {
    exec: E,
    params: FixedParams,
}

impl<E: Exec> Fixed<E> {
    pub fn new(exec: E, params: FixedParams) -> Self {
        assert!(
            params.initial_cash >= 0.0,
            "trader::fixed: initial_cash must be non-negative, got {}",
            params.initial_cash
        );
        assert!(
            params.fee_base_buy >= 0.0,
            "trader::fixed: fee_base_buy must be non-negative, got {}",
            params.fee_base_buy
        );
        assert!(
            params.fee_base_sell >= 0.0,
            "trader::fixed: fee_base_sell must be non-negative, got {}",
            params.fee_base_sell
        );
        assert!(
            params.fee_rate_buy >= 0.0,
            "trader::fixed: fee_rate_buy must be non-negative, got {}",
            params.fee_rate_buy
        );
        assert!(
            params.fee_rate_sell >= 0.0,
            "trader::fixed: fee_rate_sell must be non-negative, got {}",
            params.fee_rate_sell
        );
        Self { exec, params }
    }
}

/// Runtime state for fixed-price traders.
pub struct FixedState<E: Exec> {
    exec: E,
    params: FixedParams,
    need_rebalance: bool,
    target_weights: Vec<f64>,
    marks: Vec<f64>,
    positions: Vec<f64>,
    cash: f64,
    net_value: f64,
}

impl<E: Exec> Operator for Fixed<E> {
    type Inputs = (
        (
            SignalPort<0>,
            ArrayPort<bool, 1>,
            ArrayPort<f64, 1>,
            ArrayPort<f64, 1>,
        ),
        (SignalPort<1>, ArrayPort<f64, 1>, ArrayPort<f64, 1>),
        (SignalPort<0>, ArrayPort<f64, 1>),
    );
    type Outputs = (ArrayPort<f64, 1>, ArrayPort<f64, 0>, ArrayPort<f64, 0>);
    type Context = Instant;
    type State = FixedState<E>;

    fn init(
        self,
        (
            (_price_signal, flags, bids, asks),
            (div_signals, share_divs, cash_divs),
            (_rebalance_signal, target_weights),
        ): <Self::Inputs as Interface>::Values<'_>,
    ) -> Self::State {
        // Check that all input arrays have the same length.
        let n = target_weights.extents()[0];
        assert_eq!(flags.extents()[0], n);
        assert_eq!(bids.extents()[0], n);
        assert_eq!(asks.extents()[0], n);
        assert_eq!(div_signals.extents()[0], n);
        assert_eq!(share_divs.extents()[0], n);
        assert_eq!(cash_divs.extents()[0], n);

        // Initialize the state with empty positions.
        let cash = self.params.initial_cash;
        FixedState {
            exec: self.exec,
            params: self.params,
            need_rebalance: false,
            target_weights: vec![0.0; n],
            marks: vec![0.0; n],
            positions: vec![0.0; n],
            cash,
            net_value: cash,
        }
    }

    fn reset<'a, 'b: 'a>(
        _: <Self::Inputs as Interface>::Values<'a>,
        s: &'b mut Self::State,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        (
            ArrayView::from_slice([s.positions.len()], &s.positions),
            ArrayView::scalar(&s.cash),
            ArrayView::scalar(&s.net_value),
        )
    }

    fn compute<'a, 'b: 'a>(
        (
            (price_signal, flags, bids, asks),
            (div_signals, share_divs, cash_divs),
            (rebalance_signal, target_weights),
        ): <Self::Inputs as Interface>::Values<'a>,
        s: &'b mut Self::State,
        _: &Self::Context,
    ) -> <Self::Outputs as Interface>::Values<'a> {
        let n = target_weights.extents()[0];

        // Check for rebalancing (immediate).
        if *rebalance_signal && !s.params.delayed {
            s.need_rebalance = true;
            for (src, dst) in target_weights.iter().zip(s.target_weights.iter_mut()) {
                *dst = *src;
            }
        }

        // Check for dividends.
        for i in 0..n {
            if div_signals[[i]] {
                assert!(
                    share_divs[[i]].is_finite() && cash_divs[[i]].is_finite(),
                    "trader::fixed: dividends must be finite for stock {i}"
                );
                s.cash += s.positions[i] * cash_divs[[i]];
                s.positions[i] *= 1.0 + share_divs[[i]];
            }
        }

        // Check for price updates, optionally rebalance.
        if *price_signal {
            for i in 0..n {
                // Update marking prices for assets with valid quotes, and set
                // marking prices to 0 for user-excluded (e.g. delisted) ones.
                if flags[[i]] {
                    assert!(
                        !bids[[i]].is_nan() && !asks[[i]].is_nan(),
                        "trader::fixed: quotes must not be NaN for stock {i}"
                    );
                    assert!(
                        bids[[i]] <= asks[[i]],
                        "trader::fixed: bid must be less than or equal to ask for stock {i}"
                    );
                    if bids[[i]].is_finite() {
                        s.marks[i] = bids[[i]];
                    } else if asks[[i]].is_finite() {
                        s.marks[i] = asks[[i]];
                    }
                } else {
                    s.marks[i] = 0.0;
                }
            }
            if s.need_rebalance {
                // Rebalance the liquid value according to the target weights.
                s.need_rebalance = false;
                let bids = bids.to_contiguous();
                let asks = asks.to_contiguous();
                let mut liquid_value = s.cash;
                for (i, &position) in s.positions.iter().enumerate() {
                    if flags[[i]] && bids[i].is_finite() {
                        liquid_value += position * bids[i];
                    }
                }
                let mut target_values = vec![0.0; n];
                for (value, &weight) in target_values.iter_mut().zip(s.target_weights.iter()) {
                    *value = weight * liquid_value;
                }
                for (i, delta) in s
                    .exec
                    .orders(&bids, &asks, &s.positions, s.cash, &target_values)
                {
                    if flags[[i]]
                        && (delta > 0.0 && asks[i].is_finite()
                            || delta < 0.0 && bids[i].is_finite())
                    {
                        let is_buy = delta > 0.0;
                        let (trade_amount, fee_base, fee_rate) = if is_buy {
                            (
                                delta * asks[i],
                                s.params.fee_base_buy,
                                s.params.fee_rate_buy,
                            )
                        } else {
                            (
                                delta * bids[i],
                                s.params.fee_base_sell,
                                s.params.fee_rate_sell,
                            )
                        };
                        let fee = (trade_amount.abs() * fee_rate).max(fee_base);
                        s.cash -= trade_amount + fee;
                        s.positions[i] += delta;
                    }
                }
            }
        }

        // Check for rebalancing (delayed).
        if *rebalance_signal && s.params.delayed {
            s.need_rebalance = true;
            for (src, dst) in target_weights.iter().zip(s.target_weights.iter_mut()) {
                *dst = *src;
            }
        }

        // Update net asset value.
        s.net_value = s.cash;
        for (&position, &mark) in s.positions.iter().zip(s.marks.iter()) {
            s.net_value += position * mark;
        }

        (
            ArrayView::from_slice([s.positions.len()], &s.positions),
            ArrayView::scalar(&s.cash),
            ArrayView::scalar(&s.net_value),
        )
    }
}
