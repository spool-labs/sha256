//! Links Firedancer and isa-l_crypto for the comparison benchmarks, when asked.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=FD_LIB_DIR");
    println!("cargo:rerun-if-env-changed=ISAL_DIR");
    println!("cargo:rerun-if-changed=benches/isal_shim.c");

    if env::var_os("CARGO_FEATURE_FIREDANCER_BENCH").is_some() {
        firedancer();
    }
    if env::var_os("CARGO_FEATURE_ISAL_BENCH").is_some() {
        isal();
    }
}

fn firedancer() {
    let Some(dir) = env::var_os("FD_LIB_DIR") else {
        panic!(
            "feature `firedancer-bench` is enabled but FD_LIB_DIR is unset.\n\
             Point it at a Firedancer build's lib directory, e.g.\n  \
             FD_LIB_DIR=/path/to/firedancer/build/native/gcc/lib"
        );
    };

    println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    // fd_ballet holds the hash implementations; fd_util holds the shared
    // runtime symbols they depend on.
    println!("cargo:rustc-link-lib=static=fd_ballet");
    println!("cargo:rustc-link-lib=static=fd_util");
}

/// Compiles the isa-l_crypto shim (benchmark) and links the static library.
///
/// `cc` is driven directly rather than through the `cc` crate: this is the
/// only C in the tree.
fn isal() {
    let Some(dir) = env::var_os("ISAL_DIR") else {
        panic!(
            "feature `isal-bench` is enabled but ISAL_DIR is unset.\n\
             Point it at an isa-l_crypto source tree that has been built \
             in place, e.g.\n  \
             ISAL_DIR=/path/to/isa-l_crypto"
        );
    };
    let dir = PathBuf::from(dir);
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    // A built-in-place tree keeps headers under include/ and the archive under
    // .libs/; an installed prefix has include/ and lib/. Accept either.
    let lib_dir = [dir.join(".libs"), dir.join("lib")]
        .into_iter()
        .find(|d| d.join("libisal_crypto.a").is_file())
        .unwrap_or_else(|| {
            panic!(
                "no libisal_crypto.a under {}/.libs or {}/lib -- run \
                 ./autogen.sh && ./configure && make in the isa-l_crypto tree",
                dir.display(),
                dir.display()
            )
        });

    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let obj = out.join("isal_shim.o");
    let status = Command::new(&cc)
        .args(["-O2", "-fPIC", "-c", "benches/isal_shim.c", "-o"])
        .arg(&obj)
        .arg("-I")
        .arg(dir.join("include"))
        .status()
        .unwrap_or_else(|e| panic!("failed to run {cc}: {e}"));
    assert!(status.success(), "compiling benches/isal_shim.c failed");

    let ar = env::var("AR").unwrap_or_else(|_| "ar".into());
    let archive = out.join("libisal_shim.a");
    let _ = std::fs::remove_file(&archive);
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to run {ar}: {e}"));
    assert!(status.success(), "archiving isal_shim.o failed");

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=isal_shim");
    println!("cargo:rustc-link-lib=static=isal_crypto");
}
