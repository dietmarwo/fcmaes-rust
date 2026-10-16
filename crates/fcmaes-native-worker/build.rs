use std::process::Command;
fn capture(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().replace('\n', " "))
        .unwrap_or_else(|| "unknown".into())
}
fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!(
        "cargo:rustc-env=FCMAES_SOURCE_COMMIT={}",
        capture("git", &["rev-parse", "HEAD"])
    );
    println!(
        "cargo:rustc-env=FCMAES_RUSTC={}",
        capture(
            &std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()),
            &["--version"]
        )
    );
    println!(
        "cargo:rustc-env=FCMAES_TARGET={}",
        std::env::var("TARGET").unwrap()
    );
}
