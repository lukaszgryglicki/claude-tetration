//! End-to-end CLI tests: invoke the `tet` binary as a subprocess and verify
//! that stdout / stderr / exit codes behave as documented.

use std::path::PathBuf;
use std::process::{Command, Output};

fn binary_path() -> PathBuf {
    // Cargo sets CARGO_BIN_EXE_<name> for tests of binaries in this crate.
    PathBuf::from(env!("CARGO_BIN_EXE_tet"))
}

fn run(args: &[&str]) -> Output {
    Command::new(binary_path())
        .args(args)
        .output()
        .expect("failed to spawn tet binary")
}

fn parse_two_lines(stdout: &[u8]) -> (String, String) {
    let s = String::from_utf8_lossy(stdout);
    let mut lines = s.lines();
    let re = lines.next().unwrap_or("").to_string();
    let im = lines.next().unwrap_or("").to_string();
    (re, im)
}

#[test]
fn t700_help_flag() {
    let out = run(&["--help"]);
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("Usage: tet"), "help output missing usage: {}", s);
}

#[test]
fn t701_wrong_arg_count_exit_code_2() {
    let out = run(&["50", "2", "0"]);
    assert_eq!(out.status.code(), Some(2));
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(s.contains("Usage"), "stderr missing usage hint: {}", s);
}

#[test]
fn t702_integer_height_two_to_three() {
    // 2^^3 = 2^(2^2) = 2^4 = 16.
    let out = run(&["50", "2", "0", "3", "0"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let (re, im) = parse_two_lines(&out.stdout);
    assert!(re.starts_with("16"), "expected 16.*, got {}", re);
    assert_eq!(im.trim_start_matches('-').trim_start_matches('0').trim_start_matches('.').trim_start_matches('0'), "");
}

#[test]
fn t703_base_one_returns_one() {
    let out = run(&["30", "1", "0", "5.5", "0.3"]);
    assert!(out.status.success());
    let (re, im) = parse_two_lines(&out.stdout);
    assert!(re.starts_with('1'), "expected 1.*, got {}", re);
    let im_zero = im.replace(['0', '-', '.'], "").is_empty();
    assert!(im_zero, "expected im≈0, got {}", im);
}

#[test]
fn t704_zero_arg_height_zero_is_one() {
    // F_b(0) = 1 for any b ≠ 1, including b = 0 (convention).
    let out = run(&["20", "0", "0", "0", "0"]);
    assert!(out.status.success());
    let (re, _im) = parse_two_lines(&out.stdout);
    assert!(re.starts_with('1'), "expected 1.*, got {}", re);
}

#[test]
fn t705_invalid_precision_errors() {
    let out = run(&["abc", "2", "0", "3", "0"]);
    assert!(!out.status.success());
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(s.contains("invalid precision"), "stderr: {}", s);
}

#[test]
fn t706_zero_precision_errors() {
    let out = run(&["0", "2", "0", "3", "0"]);
    assert!(!out.status.success());
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(s.contains("precision"), "stderr: {}", s);
}

#[test]
fn t707_schroder_path_cli() {
    // b = √2 (interior), h = 0.5 — Schröder, no warning.
    let out = run(&["30", "1.4142135623730950488", "0", "0.5", "0"]);
    assert!(out.status.success());
    let (re, _im) = parse_two_lines(&out.stdout);
    // F_{√2}(0.5) ≈ 1.24362... (cross-check with Phase 4 tests).
    assert!(re.starts_with("1.243"), "got {}", re);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("warning"), "unexpected warning: {}", stderr);
}

#[test]
fn t708_real_e_via_kouznetsov_cli() {
    // b = e at non-integer height — Schröder's σ̃-shift can't reach 1−L from
    // L for this base, so the dispatcher routes to the Newton-Kantorovich
    // Kouznetsov path. Verify the CLI exits success and the value is in the
    // ballpark of the published Kneser tetration value `e^^0.5 ≈ 1.6463`.
    let out = run(&["20", "2.71828182845904523536", "0", "0.5", "0"]);
    assert!(
        out.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let re = stdout.lines().next().unwrap_or("");
    assert!(re.starts_with("1.6"), "got {}", re);
}

#[test]
fn t709_debug_diagnostics() {
    // Verbose-by-default: stderr should contain `tet: …` diagnostics when SILENT
    // is unset (or falsy). Setting SILENT=1 should suppress all stderr output
    // and produce only the 2-line numeric result on stdout.
    let out_verbose = Command::new(binary_path())
        .args(["20", "1.4142135623730950488", "0", "0.5", "0"])
        .env_remove("SILENT")
        .output()
        .expect("spawn");
    assert!(out_verbose.status.success());
    let stderr = String::from_utf8_lossy(&out_verbose.stderr);
    assert!(stderr.contains("tet:"), "expected default verbose output, got: {}", stderr);
    assert!(stderr.contains("region = "), "expected region in verbose output, got: {}", stderr);

    let out_silent = Command::new(binary_path())
        .args(["20", "1.4142135623730950488", "0", "0.5", "0"])
        .env("SILENT", "1")
        .output()
        .expect("spawn");
    assert!(out_silent.status.success());
    let silent_err = String::from_utf8_lossy(&out_silent.stderr);
    assert!(
        silent_err.is_empty(),
        "expected no stderr under SILENT=1, got: {}",
        silent_err
    );
}

#[test]
fn t710_boundary_band_iperturbation_fallback() {
    // Base on the parabolic Shell-Thron boundary band (|λ| ≈ 1) — Schröder is
    // unreliable there and direct/continuation Kouznetsov hit the parabolic
    // n_nodes cap. The dispatcher falls back to an iε-perturbation Richardson
    // extrapolation: it tetrates at b+0.1i and b+0.05i (which take the
    // OutsideShellThronGeneral / Kouznetsov path successfully) and combines
    // them as (4·F(ε/2) − F(ε))/3 to cancel the O(ε²) leading error of the
    // even part, with the odd imaginary part collapsing to zero. Practical
    // reach is roughly 6 digits, far short of the requested precision but
    // enough for the CLI to deliver a real-valued answer instead of failing.
    // b ≈ η = e^(1/e) ≈ 1.4446679 is the canonical parabolic point.
    let out = run(&["20", "1.444667861009766", "0", "0.5", "0"]);
    assert!(out.status.success(), "expected success, got exit={:?}, stderr={}",
            out.status.code(), String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("iε-perturbation") || stderr.contains("parabolic-boundary fallback"),
        "expected iε-perturbation warning, stderr: {}",
        stderr
    );
    // Output should be a real tetration value near 1.257 (the empirical
    // e^^(0.5) limit value at b=η).
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "expected re/im output, got: {}", stdout);
    let re: f64 = lines[0].parse().expect("re parse");
    let im: f64 = lines[1].parse().expect("im parse");
    assert!(
        (re - 1.257).abs() < 0.01,
        "expected Re ≈ 1.257, got {}",
        re
    );
    assert!(
        im.abs() < 1e-10,
        "expected Im ≈ 0 (Schwarz cancellation), got {}",
        im
    );
}
