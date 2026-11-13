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

/// A name or an expression is its number to the digits of the run: `b^^1 = b`.
#[test]
fn t712_constants_and_expressions_have_the_requested_digits() {
    let pi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303820";
    let eta = "1.44466786100976613365833910859643022305859545324225316582052";
    let towermin = "6.59880358453125370767901875968464249385770482527964364024735e-2";
    for (digits, base, expected) in [
        ("200", "pi", pi),
        (
            "60",
            "e",
            "2.71828182845904523536028747135266249775724709369995957496697",
        ),
        ("60", "eta", eta),
        ("60", "e^(1/e)", eta),
        ("60", "towermin", towermin),
        ("60", "e^-e", towermin),
        ("20", "-pi", "-3.1415926535897932385"),
        (
            "30",
            "pi*e^e-pow(2, 0.5)",
            "46.1943053660313545855098891412",
        ),
    ] {
        let out = run(&["-q", digits, base, "0", "1", "0"]);
        assert!(out.status.success(), "{base}: {out:?}");
        assert!(out.stderr.is_empty(), "{base}: {out:?}");
        let (re, im) = parse_two_lines(&out.stdout);
        assert_eq!((re.as_str(), im.as_str()), (expected, "0"), "{base}");
    }
}

/// The number is `first + i second`, each of them may be complex, and a
/// tetration inside an argument is computed by the same library.
#[test]
fn t713_complex_expressions_and_tetration_inside_arguments() {
    let value = |arguments: &[&str]| {
        let out = run(arguments);
        assert!(out.status.success(), "{arguments:?}: {out:?}");
        parse_two_lines(&out.stdout)
    };
    let direct = value(&["-q", "20", "2", "1", "1.5", "0.5"]);
    for (first, second) in [("2+i", "0"), ("i", "-2i"), ("2", "sqrt(1)"), ("0", "1-2i")] {
        let same = value(&["-q", "20", first, second, "3/2", "1/2"]);
        assert_eq!(same, direct, "{first} {second}");
    }
    let tower = value(&["-q", "20", "2^^2", "0", "tet(2, 1)", "0"]);
    assert_eq!(tower.0, "256.00000000000000000");
    // The inverses in the arguments: the base 2 and the height 3.
    let sixteen = value(&["-q", "15", "ssrt(16, 3)", "0", "ate(2, 16)", "0"]);
    assert_eq!(sixteen.0, "16.0000000000000");
    let height = value(&["-q", "--slog", "15", "sqrt(2)^^2 / 1", "0", "2^^3", "0"]);
    let plain = value(&[
        "-q",
        "--slog",
        "15",
        "1.632526919438152844773495",
        "0",
        "16",
        "0",
    ]);
    assert_eq!(height, plain);
    // A whole height keeps its parity however large it is: 0^^n is 1, 0, 1, ...
    let one = "1.00000000000000";
    for (height, expected) in [("10^100", one), ("10^100 + 1", "0"), ("2*3", one)] {
        let zero = value(&["-q", "15", "0", "0", height, "0"]);
        assert_eq!(zero.0, expected, "{height}");
    }
    for (argument, reason) in [
        ("x", "unknown name x"),
        ("sin(pi)", "cancels below every precision"),
        ("1/0", "division by zero"),
        ("2^^", "ends early"),
    ] {
        let out = run(&["20", argument, "0", "1", "0"]);
        assert_eq!(out.status.code(), Some(1), "{argument}");
        assert!(out.stdout.is_empty(), "{argument}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains("invalid number"), "{argument}: {stderr}");
        assert!(stderr.contains(reason), "{argument}: {stderr}");
    }
}

/// `--help`, the usage printed without arguments and the README name every
/// constant and every function.
#[test]
fn t714_help_and_readme_list_every_constant_and_function() {
    let help = String::from_utf8(run(&["--help"]).stdout).unwrap();
    let bare = run(&[]);
    assert_eq!(bare.status.code(), Some(2));
    assert_eq!(String::from_utf8(bare.stderr).unwrap(), help);
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("README.md");
    let listed = |names: &[&str], meaning: &str| {
        let line = format!("    {} ", names.join(", "));
        assert!(
            help.lines()
                .any(|found| found.starts_with(&line) && found.ends_with(meaning)),
            "--help lacks {names:?}"
        );
        for name in names {
            assert!(readme.contains(&format!("`{name}`")), "README lacks {name}");
        }
    };
    for named in tetration::constants::TABLE {
        let mut names = vec![named.name];
        names.extend(named.aliases);
        listed(&names, named.meaning);
    }
    for function in tetration::expression::FUNCTIONS {
        listed(function.names, function.meaning);
    }
}
