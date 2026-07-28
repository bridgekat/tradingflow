//! Performance metric operators on net-asset-value scalars.

mod drawdown;
mod r#return;
mod turnover;

pub use drawdown::drawdown;
pub use r#return::{
    comp_return, log_return_mean, log_return_sharpe, log_return_vol, return_mean, return_sharpe,
    return_vol,
};
pub use turnover::turnover;
