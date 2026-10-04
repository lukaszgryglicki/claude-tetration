use rug::Complex;
use tetration::{cnum, lambertw};

fn main() {
    let prec = cnum::digits_to_bits(8);
    // -ln(3+1i)
    let three_plus_i = cnum::parse_complex("3", "1", prec).unwrap();
    let ln_b = Complex::with_val_64(prec, three_plus_i.ln_ref());
    let neg_ln_b = Complex::with_val_64(prec, -&ln_b);
    eprintln!(
        "z = -ln(3+i) = {:.7e} + {:.7e}i",
        cnum::DisplayFloat(neg_ln_b.real()),
        cnum::DisplayFloat(neg_ln_b.imag())
    );
    match lambertw::w0(&neg_ln_b, prec) {
        Ok(w) => {
            eprintln!(
                "W_0(z) = {:.7e} + {:.7e}i",
                cnum::DisplayFloat(w.real()),
                cnum::DisplayFloat(w.imag())
            );
        }
        Err(e) => eprintln!("error: {}", e),
    }
}
