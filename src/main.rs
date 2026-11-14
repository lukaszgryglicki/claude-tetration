use std::env;
use std::process::ExitCode;

const USAGE: &str = "\
Usage: tet [--quiet|--silent|-q] <precision_digits> <base_re> <base_im> <height_re> <height_im>
       tet [-q] --slog|--ate   <precision_digits> <base_re> <base_im> <value_re> <value_im>
       tet [-q] --sroot|--ssrt <precision_digits> <value_re> <value_im> <height_re> <height_im>
       tet [-q] <precision_digits> <base> <height>
       tet [-q] --slog|--ate   <precision_digits> <base> <value>
       tet [-q] --sroot|--ssrt <precision_digits> <value> <height>

  Computes complex tetration F_b(h), where F_b(0) = 1 and F_b(z+1) = b^F_b(z).
  The three modes concern one equation, b^^h = value (that is, F_b(h) = value):
  give two of base, height and value, and the third is printed. Each of the
  two is written as two real numbers, its real part then its imaginary part,
  or as one complex number: tet 30 1.5 0.5 1 0.7 is tet 30 1.5+0.5i 1+0.7i.

    mode               arguments (after digits)   prints                   like
    no flag            base b, height h           value b^^h               b^h
    --slog  | --ate    base b, value              height h: b^^h = value   log_b(value)
    --sroot | --ssrt   value, height h            base b: b^^h = value     value^(1/h)

  --slog  (alias --ate, arc-tetration): super-logarithm.
  --sroot (alias --ssrt): super-root; the value comes first, the height second.
  Both verify their answer by computing b^^h again, and fail when there is no
  solution they can find and verify. The flags may stand anywhere on the
  command line.

Arguments:
  precision_digits     positive integer; number of decimal digits of precision
  base_re, base_im     the base b = base_re + i base_im
  height_re, height_im the height h = height_re + i height_im
  value_re, value_im   the tetration value to invert, value_re + i value_im
  base, height, value  the same three numbers, each as one complex argument

  Each of these is a decimal number or a constant expression (quote it
  for the shell), computed to the precision of the run:

    numbers      12, -0.5, 1e-30, and i (or j) for the imaginary unit
    constants    the names listed below: pi, e, eta, ...
    operators    + - * / ^ (or **), ^^ for tetration, and brackets
    products     need no sign: 2pi, 3i, 2(1+i)
    functions    the names listed below: sqrt(2), ln(3), tet(b, h), ...

  A leading sign binds less than ^, and ^ less than ^^; both group from the
  right: -2^2 = -4, 2^3^2 = 2^9, 2^^3^2 = (2^^3)^2. There are no variables.
  With four numbers each of them must be real, a complex one is an error:
  re(z) is the real part of anything. With two numbers each is complex, so
  2 1 0.5 0 and 2+i 0.5 are the same base and height.

  The expression is evaluated twice, with every operation rounded down and
  with every operation rounded up, until both agree beyond the precision of
  the run; digits lost to cancellation are added. A value that cancels to
  nothing at every precision tried is refused, like sin(pi): write 0. A
  tetration inside an argument is computed by tet itself, to more digits
  than the run's.

Constants (names, meaning):
{constants}

Functions (z, w complex; x, y real; names with the same meaning share a line):
{functions}

Output (on stdout):
  Two lines: real part on line 1, imaginary part on line 2, of the value
  (of the height with --slog, of the base with --sroot).

Diagnostics:
  Algorithm choice and per-iteration diagnostics print to stderr by
  default. Use --quiet, --silent or -q, or set SILENT=1 (also
  t/true/y/yes/on), to suppress progress diagnostics. Fatal errors
  still print to stderr and exit with nonzero status.

Examples:
  tet 50 2 0 3 0          # 2^^3 = 16, 50 digits
  tet 50 2 3              # the same, each number as one argument
  tet 100 e 0 0.5 0       # base e to 100 digits
  tet 30 1.5 0.5 1 0.7    # complex base, complex height
  tet 30 1.5+0.5i 1+0.7i  # the same
  tet 30 'pi*e^e-pow(2, 0.5)' 0 '1/3' 0
  tet 30 '2^^2' 're(ate(2+i, 1.5-0.5i))' 'pi/e' 'ssrt(16, 3)/4'
  tet 30 '2^^2 + i*ate(2+i, 1.5-0.5i)' 'pi/e + i*ssrt(16, 3)/4'
  tet --slog 30 2 0 16 0  # the height h with 2^^h = 16, i.e. 3
  tet --slog 30 2 16      # the same
  tet --sroot 30 16 0 3 0 # the base b with b^^3 = 16, i.e. 2
  tet --sroot 30 16 3     # the same
";

fn usage() -> String {
    USAGE
        .replace("{constants}\n", &tetration::constants::help())
        .replace("{functions}\n", &tetration::expression::help())
}

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
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }
    let result = match &args[1..] {
        _ if slog && sroot => None,
        [digits, first, second] => Some(if slog {
            tetration::slog_complex_str(digits, first, second)
        } else if sroot {
            tetration::sroot_complex_str(digits, first, second)
        } else {
            tetration::tetrate_complex_str(digits, first, second)
        }),
        [digits, first_re, first_im, second_re, second_im] => Some(if slog {
            tetration::slog_str(digits, first_re, first_im, second_re, second_im)
        } else if sroot {
            tetration::sroot_str(digits, first_re, first_im, second_re, second_im)
        } else {
            tetration::tetrate_str(digits, first_re, first_im, second_re, second_im)
        }),
        _ => None,
    };
    let Some(result) = result else {
        eprint!("{}", usage());
        return ExitCode::from(2);
    };
    match result {
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
