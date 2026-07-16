mod flat;
mod graph;
mod handles;
mod interface;
mod operator;
mod segment;
mod source;

pub use flat::{FlatRead, FlatWrite};
pub use graph::{Builder, Graph};
pub use handles::{Handle, HandlesInterface, InterfaceHandles, OutputHandle, SourceHandle};
pub use interface::{
    Interface, Port, Ports, RefPort, RefPorts, RefViewPort, RefViewPorts, Scalar, Slice, ValueView,
    ViewPort, ViewPorts,
};
pub use operator::Operator;
pub use segment::{Arr, Comp, Fork, Id, Left, Par, Right, Segment, SegmentExt};
pub use source::{RefSource, RefViewSource, Source, ViewSource};
