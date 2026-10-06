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
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (re, im) = parse_two_lines(&out.stdout);
    assert!(re.starts_with("16"), "expected 16.*, got {}", re);
    assert_eq!(
        im.trim_start_matches('-')
            .trim_start_matches('0')
            .trim_start_matches('.')
            .trim_start_matches('0'),
        ""
    );
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
    assert!(
        !stderr.contains("warning"),
        "unexpected warning: {}",
        stderr
    );
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
    assert!(
        stderr.contains("tet:"),
        "expected default verbose output, got: {}",
        stderr
    );
    assert!(
        stderr.contains("region = "),
        "expected region in verbose output, got: {}",
        stderr
    );

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
#[ignore = "Very close strict attractor: normalization can take hours; routine fringe coverage is in phase12"]
fn t710_very_close_attracting_boundary_preserves_cli_digits() {
    let mut values = Vec::new();
    for digits in ["20", "40"] {
        let out = run(&["--quiet", digits, "1.444667861009766", "0", "0.5", "0"]);
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.stderr.is_empty());
        let (re, im) = parse_two_lines(&out.stdout);
        assert_eq!(im, "0");
        values.push(re);
    }
    let reference =
        tetration::cnum::parse_float(&values[1], tetration::cnum::digits_to_bits(70)).unwrap();
    assert_eq!(values[0], tetration::cnum::format_float(&reference, 20));
}

#[test]
fn t711_long_decimal_inputs_preserve_exact_domains() {
    for digits in [1, 10, 50, 70, 1000] {
        let zeros = "0".repeat(digits + 100);
        let precision = digits.to_string();
        let near_one = format!("1.{zeros}1");
        let odd = format!("1{zeros}1");
        let fractional = format!("2.{zeros}1");
        for (base, height, valid) in [
            (near_one.as_str(), "-1", true),
            ("0", odd.as_str(), true),
            ("0", fractional.as_str(), false),
        ] {
            let mut serial_stdout = None;
            for parallel in [false, true] {
                let mut command = Command::new(binary_path());
                command.args(["--quiet", &precision, base, "0", height, "0"]);
                if parallel {
                    command.env("TET_MT", "4").env("RAYON_NUM_THREADS", "4");
                } else {
                    command.env_remove("TET_MT").env_remove("RAYON_NUM_THREADS");
                }
                let out = command.output().expect("spawn");
                if valid {
                    assert!(out.status.success(), "{out:?}");
                    assert_eq!(out.stdout, b"0\n0\n");
                    assert!(out.stderr.is_empty(), "{out:?}");
                } else {
                    assert_eq!(out.status.code(), Some(1), "{out:?}");
                    assert!(out.stdout.is_empty(), "{out:?}");
                    assert!(
                        String::from_utf8_lossy(&out.stderr).contains(
                            "tetration of 0 is only defined for non-negative integer heights"
                        ),
                        "{out:?}"
                    );
                }
                if let Some(serial) = &serial_stdout {
                    assert_eq!(&out.stdout, serial);
                } else {
                    serial_stdout = Some(out.stdout);
                }
            }
        }
    }
}
