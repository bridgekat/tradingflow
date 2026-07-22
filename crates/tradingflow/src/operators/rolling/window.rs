use crate::data::Duration;

/// Rolling window selection strategy.
#[derive(Debug, Clone, Copy)]
pub enum Window {
    Count(usize),
    TimeDelta(Duration),
}
