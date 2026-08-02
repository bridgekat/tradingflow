//! Startup of the embedded interpreter.
//!
//! # Why this exists
//!
//! An embedded interpreter has no `python` executable to take its bearings
//! from. It is a Rust binary that happens to link `libpython`, so CPython's
//! usual path computation — locate the running executable, read a `pyvenv.cfg`
//! beside it, otherwise search for landmarks like `Lib/os.py` — starts from
//! wherever the binary sits (`target/debug/deps/`, say) and finds nothing.
//! Having no `sys.prefix`, the interpreter cannot import `encodings`, and it
//! aborts the process before any Rust error handling gets a turn:
//!
//! ```text
//! Could not find platform independent libraries <prefix>
//! Fatal Python error: Failed to import encodings module
//! ```
//!
//! Pointing `PYTHONHOME` at an installation gets past that, but at the cost of
//! the environment's `site-packages` — the interpreter then finds its standard
//! library and nothing else, so `import numpy` fails — which is what makes
//! `PYTHONPATH` necessary to reinstate the packages by hand.
//!
//! # What it does instead
//!
//! [`initialize`] tells CPython which interpreter it is standing in for, by
//! setting `PyConfig.executable` to the one PyO3 resolved and linked against
//! (recorded at build time by `build.rs`, so it cannot disagree with the
//! `libpython` the binary is bound to). CPython then does everything it would
//! have done had that interpreter been run directly: if it names a virtual
//! environment, read its `pyvenv.cfg`, take the standard library from the base
//! installation it points at, and put the environment's `site-packages` on
//! `sys.path` — `.pth` files and all, so an editable install (`uv sync`, `pip
//! install -e .`) of the operator package is importable as itself.
//!
//! No `PYTHONHOME`, no `PYTHONPATH`, and nothing for a caller to set up. Both
//! variables still work if set, for anyone who wants to override the
//! environment the build picked.

use std::sync::Once;

use libc::wchar_t;
use pyo3::ffi;
use pyo3::prelude::*;

/// Absolute path of the interpreter PyO3 linked against, from `build.rs`.
///
/// `None` when the build could not name one, in which case startup is left to
/// CPython's own path computation — the behaviour before this module existed.
const EXECUTABLE: Option<&str> = option_env!("TRADINGFLOW_PYTHON_EXECUTABLE");

static START: Once = Once::new();

/// Runs `f` with the interpreter attached, starting it if necessary.
///
/// A drop-in for [`Python::attach`] that cannot be reached before startup.
pub fn attach<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(Python<'py>) -> R,
{
    initialize();
    Python::attach(f)
}

/// Starts the embedded interpreter, unless one is already running.
///
/// Idempotent, and cheap after the first call. Everything in this module
/// reaches Python through [`attach`], which calls this first, so there is
/// normally no reason to call it directly; an embedder that reaches for
/// `pyo3` itself must call it before its own first `Python::attach`.
///
/// # Panics
///
/// Exits the process if CPython cannot start, the only thing a failed
/// interpreter startup permits — there is no interpreter left to raise with.
pub fn initialize() {
    START.call_once(|| {
        // Someone else may have started an interpreter already (an embedder,
        // or an extension module hosting us); it is theirs to configure.
        // SAFETY: `Py_IsInitialized` is callable without an interpreter — that
        // is the question it answers.
        if unsafe { ffi::Py_IsInitialized() } == 0
            && let Some(executable) = EXECUTABLE
        {
            // SAFETY: no interpreter is running, as just checked, and `Once`
            // makes this the only thread that can be here.
            unsafe { initialize_as(executable) };
        }
    });
    // Let PyO3 note the interpreter as started. It skips its own
    // `Py_InitializeEx` when one is already running, so this either adopts
    // what the block above built or, if `EXECUTABLE` was `None`, performs the
    // default startup.
    Python::initialize();
}

/// Starts CPython as though it had been launched as `executable`.
///
/// # Safety
///
/// No interpreter may be running, and no other thread may be starting one.
unsafe fn initialize_as(executable: &str) {
    unsafe {
        let mut config = std::mem::zeroed::<ffi::PyConfig>();
        ffi::PyConfig_InitPythonConfig(&mut config);

        // Everything else — prefix, standard library, `site-packages`, and
        // whether this is a virtual environment at all — CPython derives from
        // this one path, exactly as it does when the interpreter is launched
        // from disk.
        let path = wide(executable);
        let status = ffi::PyConfig_SetString(&mut config, &mut config.executable, path.as_ptr());
        if ffi::PyStatus_Exception(status) != 0 {
            ffi::PyConfig_Clear(&mut config);
            ffi::Py_ExitStatusException(status);
        }

        let status = ffi::Py_InitializeFromConfig(&config);
        ffi::PyConfig_Clear(&mut config);
        if ffi::PyStatus_Exception(status) != 0 {
            ffi::Py_ExitStatusException(status);
        }

        // Startup leaves this thread holding the GIL. Operator nodes run on
        // the engine's worker threads, so holding it here would deadlock the
        // first attach from any of them; PyO3 acquires it per attach.
        ffi::PyEval_SaveThread();
    }
}

/// Encodes a path as the NUL-terminated `wchar_t` string `PyConfig` wants.
fn wide(path: &str) -> Vec<wchar_t> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;

        std::ffi::OsStr::new(path)
            .encode_wide()
            .map(|unit| unit as wchar_t)
            .chain([0])
            .collect()
    }
    #[cfg(not(windows))]
    {
        // Elsewhere `wchar_t` holds a code point, which is what `chars` yields.
        path.chars().map(|c| c as wchar_t).chain([0]).collect()
    }
}
