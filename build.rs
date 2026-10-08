//! Records the resolved rust_local_ai version so Python can report it.

use std::{env, fs, path::Path};

fn main() {
    let lock = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let version = fs::read_to_string(&lock)
        .ok()
        .and_then(|text| {
            let mut lines = text.lines();
            while let Some(line) = lines.next() {
                if line == "name = \"rust_local_ai\"" {
                    let next = lines.next()?;
                    return next
                        .strip_prefix("version = \"")
                        .and_then(|v| v.strip_suffix('"'))
                        .map(str::to_owned);
                }
            }
            None
        })
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=RUST_LOCAL_AI_VERSION={version}");
}
