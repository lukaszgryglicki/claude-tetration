// Temporary probe: cold Kouznetsov convergence along b = 0.04 + iy.
use rug::{Complex, Float};
use tetration::{kouznetsov, regions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let y: f64 = args[1].parse().unwrap();
    let prec: u32 = 128;
    let digits: u64 = 20;
    let b = Complex::with_val(prec, (Float::with_val(prec, 0.04), Float::with_val(prec, y)));
    let region = regions::classify(&b, prec).unwrap();
    let fp = match &region {
        regions::Region::ShellThronInterior(fp)
        | regions::Region::ShellThronBoundary(fp)
        | regions::Region::OutsideShellThronRealPositive(fp)
        | regions::Region::OutsideShellThronGeneral(fp) => fp.clone(),
        _ => panic!("special base"),
    };
    eprintln!("y={y}: |λ|={:.6}", fp.lambda_abs);
    let t0 = std::time::Instant::now();
    match kouznetsov::setup_kouznetsov(&b, &fp, prec, digits) {
        Ok(st) => {
            let h = Complex::with_val(prec, (Float::with_val(prec, 0.5), Float::new(prec)));
            match kouznetsov::eval_kouznetsov(&st, &b, &h) {
                Ok(v) => println!("y={y}: OK  F={:.20}  ({:.1}s)", v, t0.elapsed().as_secs_f64()),
                Err(e) => println!("y={y}: EVAL-ERR {e}  ({:.1}s)", t0.elapsed().as_secs_f64()),
            }
        }
        Err(e) => println!("y={y}: SETUP-ERR {e}  ({:.1}s)", t0.elapsed().as_secs_f64()),
    }
}
