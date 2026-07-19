use std::future::Future;

/// A generic clock source for the event loop.
pub trait Clock<I> {
    /// Returns current wall-clock reading. Must be non-decreasing.
    fn now(&self) -> I;

    /// Returns a future that resolves once the wall clock has moved strictly
    /// past `t` (so `now() > t` afterwards).
    fn wait_until(&mut self, instant: I) -> impl Future<Output = ()>;
}
