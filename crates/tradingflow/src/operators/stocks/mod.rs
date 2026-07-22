//! Stock-specific operators: `Annualize` (YTD → annualized) and `ForwardAdjust`
//! (corporate-action price adjustment, message-passing on price vs dividend
//! inputs), over the strided [`ArrayView`](crate::data::ArrayView) currency.

mod annualize;
mod forward_adjust;

pub use annualize::*;
pub use forward_adjust::*;
