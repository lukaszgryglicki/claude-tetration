// Probe Schröder warm samples on the Kouznetsov line Re z = 0.5 at the anchor base.
use rug::{Complex, Float};
use tetration::{lambertw, regions, schroder};

fn main() {
    let prec: u32 = 192;
    let b = Complex::with_val(prec, (Float::with_val(prec, 0.04), Float::with_val(prec, 0.15)));
    let region = regions::classify(&b, prec).unwrap();
    let fp = match &region {
        regions::Region::ShellThronInterior(fp) => fp.clone(),
        r => panic!("unexpected region {:?}", r.name()),
    };
    let ss = schroder::setup_schroder(&b, &fp, prec).unwrap();
    let ln_b = Complex::with_val(prec, b.ln_ref());
    let neg_ln_b = Complex::with_val(prec, -&ln_b);
    let wm1 = lambertw::wk(&neg_ln_b, -1, prec).unwrap();
    let l_low = Complex::with_val(prec, Complex::with_val(prec, -&wm1) / &ln_b);
    let l_up = fp.fixed_point.clone();
    eprintln!("L_up = {:.6}", l_up);
    eprintln!("L_low = {:.6}", l_low);
    for t in [8.0, 4.0, 2.0, 1.0, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0, -8.0, -12.0, -16.0, -20.0] {
        let z = Complex::with_val(prec, (Float::with_val(prec, 0.5), Float::with_val(prec, t)));
        match schroder::eval_schroder(&ss, &z) {
            Ok(v) => {
                let du = Float::with_val(prec, Complex::with_val(prec, &v - &l_up).abs_ref()).to_f64();
                let dl = Float::with_val(prec, Complex::with_val(prec, &v - &l_low).abs_ref()).to_f64();
                let vr = Float::with_val(64, v.real()).to_f64();
                let vi = Float::with_val(64, v.imag()).to_f64();
                println!("t={t:7}: F = {vr:+.6} {vi:+.6}i   |F-Lup|={du:.3e}  |F-Llow|={dl:.3e}");
            }
            Err(e) => println!("t={t:7}: ERR {e}"),
        }
    }
}
