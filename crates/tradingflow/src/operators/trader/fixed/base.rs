use crate::data::{ArrayView, Instant};
use crate::graph::{Interface, Segment};
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

/// Operator signature for fixed-price traders.
pub struct Fixed<E: Exec> {
    exec: E,
    delayed: bool,
    initial_cash: f64,
    fee_base: f64,
    fee_rate: f64,
}

impl<E: Exec> Fixed<E> {
    pub fn new(exec: E, delayed: bool, initial_cash: f64, fee_base: f64, fee_rate: f64) -> Self {
        assert!(
            initial_cash >= 0.0,
            "trader::fixed: initial_cash must be non-negative, got {initial_cash}"
        );
        assert!(
            fee_base >= 0.0,
            "trader::fixed: fee_base must be non-negative, got {fee_base}"
        );
        assert!(
            fee_rate >= 0.0,
            "trader::fixed: fee_rate must be non-negative, got {fee_rate}"
        );
        Self {
            exec,
            delayed,
            initial_cash,
            fee_base,
            fee_rate,
        }
    }
}

/// Runtime state for fixed-price traders.
pub struct FixedState<E: Exec> {
    exec: E,
    delayed: bool,
    fee_base: f64,
    fee_rate: f64,
    need_rebalance: bool,
    target_weights: Vec<f64>,
    marks: Vec<f64>,
    positions: Vec<f64>,
    cash: f64,
    net_value: f64,
}

impl<E: Exec> Segment for Fixed<E> {
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
        FixedState {
            exec: self.exec,
            delayed: self.delayed,
            fee_base: self.fee_base,
            fee_rate: self.fee_rate,
            need_rebalance: false,
            target_weights: vec![0.0; n],
            marks: vec![0.0; n],
            positions: vec![0.0; n],
            cash: self.initial_cash,
            net_value: self.initial_cash,
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
        if *rebalance_signal && !s.delayed {
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
                s.need_rebalance = false;
                let mut net_value = s.cash;
                for (&position, &mark) in s.positions.iter().zip(s.marks.iter()) {
                    net_value += position * mark;
                }
                let mut target_values = vec![0.0; n];
                for (value, &weight) in target_values.iter_mut().zip(s.target_weights.iter()) {
                    *value = weight * net_value;
                }
                let bids = bids.to_contiguous();
                let asks = asks.to_contiguous();
                for (i, delta) in s
                    .exec
                    .orders(&bids, &asks, &s.positions, s.cash, &target_values)
                {
                    if flags[[i]]
                        && (delta > 0.0 && asks[i].is_finite()
                            || delta < 0.0 && bids[i].is_finite())
                    {
                        let trade_value = -delta * if delta > 0.0 { asks[i] } else { bids[i] };
                        let fee = (trade_value.abs() * s.fee_rate).max(s.fee_base);
                        s.cash += trade_value - fee;
                        s.positions[i] += delta;
                    }
                }
            }
        }

        // Check for rebalancing (delayed).
        if *rebalance_signal && s.delayed {
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
