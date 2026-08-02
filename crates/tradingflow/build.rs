//! Records where the interpreter PyO3 links against lives.
//!
//! The embedded runtime needs to name that interpreter at startup (see
//! `src/python/interpreter.rs`). Taking the path from `pyo3-build-config` —
//! rather than an environment variable read at run time — means the path can
//! never disagree with the `libpython` the binary is actually bound to, since
//! both come from the same resolution `pyo3-ffi` already performed.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    #[cfg(feature = "python")]
    {
        // `None` if the build had no interpreter to ask (a cross build, or a
        // `PYO3_CONFIG_FILE` that omits it); startup falls back to CPython's
        // own path computation in that case.
        if let Some(executable) = pyo3_build_config::get().executable() {
            println!("cargo::rustc-env=TRADINGFLOW_PYTHON_EXECUTABLE={executable}");
        }
        println!("cargo::rerun-if-env-changed=PYO3_PYTHON");
        println!("cargo::rerun-if-env-changed=PYO3_CONFIG_FILE");
    }
}
