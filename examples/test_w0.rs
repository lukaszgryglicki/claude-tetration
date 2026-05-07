use rug::{Complex, Float};
use tetration::{cnum, lambertw};

fn main() {
    let prec = cnum::digits_to_bits(8);
    // -ln(3+1i)
    let three_plus_i = cnum::parse_complex("3", "1", prec).unwrap();
    let ln_b = Complex::with_val(prec, three_plus_i.ln_ref());
    let neg_ln_b = Complex::with_val(prec, -&ln_b);
    eprintln!("z = -ln(3+i) = {:.6e} + {:.6e}i",
        Float::with_val(prec, neg_ln_b.real()).to_f64(),
        Float::with_val(prec, neg_ln_b.imag()).to_f64()
    );
    match lambertw::w0(&neg_ln_b, prec) {
        Ok(w) => {
            eprintln!("W_0(z) = {:.6e} + {:.6e}i",
                Float::with_val(prec, w.real()).to_f64(),
                Float::with_val(prec, w.imag()).to_f64()
            );
        }
        Err(e) => eprintln!("error: {}", e),
    }
}
