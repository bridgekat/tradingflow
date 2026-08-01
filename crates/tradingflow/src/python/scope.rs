use pyo3::prelude::*;
use std::fmt;

use super::{NativeScalar, NativeView};
use crate::data::{ArrayView, SeriesView};

/// The set of [`NativeView`]s bound for one Python call.
pub struct Scope {
    views: Vec<Py<NativeView>>,
}

impl Scope {
    pub fn new() -> Self {
        Self { views: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            views: Vec::with_capacity(capacity),
        }
    }

    /// Binds an [`ArrayView`] as a rank-`N` buffer.
    ///
    /// # Safety
    ///
    /// The scope must be closed with [`close`](Self::close) before `v`'s memory
    /// is freed, moved or written to, and an `Err` from `close` must be treated
    /// as fatal.
    ///
    /// Dropping the scope instead of closing it still invalidates every view,
    /// so no *later* acquisition can reach the payload — but it skips the
    /// export check, leaving a pointer Python captured during the call
    /// undetected and, once the payload dies, dangling.
    pub unsafe fn array<'py, T: NativeScalar, const N: usize>(
        &mut self,
        py: Python<'py>,
        v: ArrayView<'_, T, N>,
    ) -> PyResult<Bound<'py, NativeView>> {
        // SAFETY: the obligation is forwarded to this method's caller.
        self.track(unsafe { NativeView::array(py, v) }?)
    }

    /// Binds a [`SeriesView`] as a `(instants, values)` pair: a rank-`N + 1`
    /// buffer whose axis 0 is time, and the `int64` nanosecond timestamps of
    /// the same window.
    ///
    /// # Safety
    ///
    /// As [`array`](Self::array), over both of `s`'s buffers.
    pub unsafe fn series<'py, T: NativeScalar, const N: usize>(
        &mut self,
        py: Python<'py>,
        s: SeriesView<'_, T, N>,
    ) -> PyResult<(Bound<'py, NativeView>, Bound<'py, NativeView>)> {
        // SAFETY: the obligation is forwarded to this method's caller.
        let instants = self.track(unsafe { NativeView::series_instants(py, s) }?)?;
        // SAFETY: as above.
        let values = self.track(unsafe { NativeView::series_values(py, s) }?)?;
        Ok((instants, values))
    }

    pub fn len(&self) -> usize {
        self.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Closes the borrow window.
    ///
    /// Returns `Err` if any view is still exported, i.e. the Python side kept a
    /// pointer into the graph past the call. The host must treat that as a
    /// contract violation and panic: nothing bad has happened *yet*, but the
    /// escaped pointer cannot be revoked, so the next generation that
    /// reallocates the payload would turn it into a use-after-free.
    ///
    /// Every view is invalidated either way, so a *view object* that was
    /// retained without being exported — harmless, since it holds no pointer —
    /// simply raises `BufferError` if it is used later.
    ///
    /// Must be called while attached to the interpreter.
    pub fn close(self, py: Python<'_>) -> Result<(), EscapedViewsError> {
        let mut escaped = self.invalidate_all();
        if !escaped.is_empty() {
            // A reference cycle created during the call can keep a consumer —
            // and so its export — alive until the collector runs, which would
            // read as an escape. Collect once and re-check, rather than paying
            // for a collection on every tick.
            let _ = py.import("gc").and_then(|gc| gc.call_method0("collect"));
            escaped = self.invalidate_all();
        }
        if escaped.is_empty() {
            Ok(())
        } else {
            Err(EscapedViewsError { escaped })
        }
    }

    fn track<'py>(&mut self, view: Bound<'py, NativeView>) -> PyResult<Bound<'py, NativeView>> {
        self.views.push(view.clone().unbind());
        Ok(view)
    }

    fn invalidate_all(&self) -> Vec<EscapedView> {
        self.views
            .iter()
            .enumerate()
            .filter_map(|(index, view)| {
                let view = view.get();
                view.invalidate().err().map(|exports| EscapedView {
                    index,
                    exports,
                    description: view.describe(),
                })
            })
            .collect()
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        for view in &self.views {
            let _ = view.get().invalidate();
        }
    }
}

#[derive(Debug, Clone)]
pub struct EscapedView {
    index: usize,
    exports: usize,
    description: String,
}

#[derive(Debug, Clone)]
pub struct EscapedViewsError {
    escaped: Vec<EscapedView>,
}

impl EscapedViewsError {
    pub fn len(&self) -> usize {
        self.escaped.len()
    }

    pub fn is_empty(&self) -> bool {
        self.escaped.is_empty()
    }
}

impl fmt::Display for EscapedViewsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} input view(s) were still exported when the call returned:",
            self.escaped.len(),
        )?;
        for view in &self.escaped {
            write!(
                f,
                "\n  - view #{} ({}): {} outstanding buffer export(s)",
                view.index, view.description, view.exports,
            )?;
        }
        write!(
            f,
            "\ninput views borrow graph memory for the duration of one call \
             only, so neither a view nor any array derived from it may be kept \
             in operator state. Copy what you need to keep, e.g. \
             `np.asarray(view).copy()`.",
        )
    }
}

impl std::error::Error for EscapedViewsError {}
