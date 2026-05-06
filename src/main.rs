use std::env;
use std::process::ExitCode;

const USAGE: &str = "\
Usage: tet <precision_digits> <base_re> <base_im> <height_re> <height_im>

  Computes complex tetration F_b(h), where F_b(0) = 1 and F_b(z+1) = b^F_b(z).

Arguments:
  precision_digits     positive integer; number of decimal digits of precision
  base_re, base_im     real / imaginary parts of base b (decimal strings)
  height_re, height_im real / imaginary parts of height h (decimal strings)

Output (on stdout):
  Two lines: real part on line 1, imaginary part on line 2.

Diagnostics:
  Set TET_DEBUG=1 in the environment to print algorithm choice and
  iteration diagnostics on stderr.

Examples:
  tet 50 2 0 3 0          # 2^^3 = 16, 50 digits
  tet 100 2.71828... 0 0.5 0
  tet 30 1.5 0.5 1 0.7    # complex base, complex height
";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print!("{}", USAGE);
        return ExitCode::SUCCESS;
    }
    if args.len() != 6 {
        eprint!("{}", USAGE);
        return ExitCode::from(2);
    }

    match tetration::tetrate_str(&args[1], &args[2], &args[3], &args[4], &args[5]) {
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
