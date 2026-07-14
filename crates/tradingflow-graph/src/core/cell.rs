use std::any::TypeId;
use std::ptr::NonNull;

#[derive(Debug)]
pub struct ErasedCell {
    data: NonNull<()>,
    type_id: TypeId,
    drop_fn: unsafe fn(p: NonNull<()>),
}

impl ErasedCell {
    #[inline]
    pub fn new<T: Send + 'static>(value: T) -> Self {
        let boxed = Box::new(value);
        ErasedCell {
            data: NonNull::new(Box::into_raw(boxed)).unwrap().cast(),
            type_id: TypeId::of::<T>(),
            drop_fn: drop_fn_for::<T>,
        }
    }

    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    #[inline]
    pub fn get(&self) -> *mut () {
        self.data.as_ptr()
    }
}

impl Drop for ErasedCell {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.data) }
    }
}

// # Safety
//
// The payload is `Send` (enforced by `new`), so the cell may move between
// threads and be dropped anywhere. A shared `&ErasedCell` exposes only the
// `TypeId` and a raw pointer -- no typed payload access -- so `Sync` does not
// require the payload to be `Sync`: every typed dereference happens in engine
// code under its exclusive-handoff scheduling (one worker per node per
// generation). Values *exposed through output slots* are shared across
// consumer threads, but those are not related to `ErasedCell` which is only
// for internal node state, never output.
unsafe impl Send for ErasedCell {}
unsafe impl Sync for ErasedCell {}

/// Drop function for a specific type `T`.
unsafe fn drop_fn_for<T: 'static>(p: NonNull<()>) {
    let value = unsafe { Box::from_raw(p.cast::<T>().as_ptr()) };
    drop(value)
}
