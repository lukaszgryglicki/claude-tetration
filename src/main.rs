use std::env;
use std::process::ExitCode;

const USAGE: &str = "\
Usage: tet [--quiet|--silent|-q] <precision_digits> <base_re> <base_im> <height_re> <height_im>
       tet [-q] --slog  <precision_digits> <base_re> <base_im> <value_re> <value_im>
       tet [-q] --sroot <precision_digits> <value_re> <value_im> <height_re> <height_im>

  Computes complex tetration F_b(h), where F_b(0) = 1 and F_b(z+1) = b^F_b(z).
  --slog  (alias --ate):  super-logarithm, the height h with F_b(h) = value.
  --sroot (alias --ssrt): super-root, the base b with F_b(h) = value.

Arguments:
  precision_digits     positive integer; number of decimal digits of precision
  base_re, base_im     real / imaginary parts of base b (decimal strings)
  height_re, height_im real / imaginary parts of height h (decimal strings)
  value_re, value_im   real / imaginary parts of the tetration value to invert

Output (on stdout):
  Two lines: real part on line 1, imaginary part on line 2.

Diagnostics:
  Algorithm choice and per-iteration diagnostics print to stderr by
  default. Use --quiet, --silent or -q, or set SILENT=1 (also
  t/true/y/yes/on), to suppress progress diagnostics. Fatal errors
  still print to stderr and exit with nonzero status.

Examples:
  tet 50 2 0 3 0          # 2^^3 = 16, 50 digits
  tet 100 2.71828... 0 0.5 0
  tet 30 1.5 0.5 1 0.7    # complex base, complex height
  tet --slog 30 2 0 16 0  # slog_2(16) = 3
  tet --sroot 30 16 0 3 0 # the base b with b^^3 = 16, i.e. 2
";

fn main() -> ExitCode {
    let mut args: Vec<String> = env::args().collect();
    let quiet = args
        .iter()
        .skip(1)
        .any(|arg| matches!(arg.as_str(), "--quiet" | "--silent" | "-q"));
    args.retain(|arg| !matches!(arg.as_str(), "--quiet" | "--silent" | "-q"));
    tetration::cnum::set_quiet(quiet);
    let slog = args
        .iter()
        .skip(1)
        .any(|arg| matches!(arg.as_str(), "--slog" | "--ate"));
    let sroot = args
        .iter()
        .skip(1)
        .any(|arg| matches!(arg.as_str(), "--sroot" | "--ssrt"));
    args.retain(|arg| !matches!(arg.as_str(), "--slog" | "--ate" | "--sroot" | "--ssrt"));

    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print!("{}", USAGE);
        return ExitCode::SUCCESS;
    }
    if args.len() != 6 || (slog && sroot) {
        eprint!("{}", USAGE);
        return ExitCode::from(2);
    }

    let compute = if slog {
        tetration::slog_str
    } else if sroot {
        tetration::sroot_str
    } else {
        tetration::tetrate_str
    };
    match compute(&args[1], &args[2], &args[3], &args[4], &args[5]) {
        Ok((re, im)) => {
            println!("{}", re);
            println!("{}", im);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}
