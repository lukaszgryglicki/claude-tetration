use rug::{Complex, Float};
use std::process::Command;
use tetration::{cnum, regions, schroder};

const BASE: &str = "0.0659880358454";
const REAL_HEIGHT_REAL: &str = "0.3678794411713974770288780288649458783328185096616548575284078154387897783032708514562437450615570203939351890772672607894583154496292936546210799065802679917737";
const REAL_HEIGHT_IMAGINARY: &str = "0.0000004449367701238916280901691001714621571400211852583624328529937859176236391192485901527770801142227611242175315729775741321507597248946444905920008822737841103301";
const COMPLEX_HEIGHT_REAL: &str = "0.3678794411714857034625286609853355930014208702866454221982309319103036459221468765888216698405053000902447217108180368945438329358975446760813405195491424049812";
const COMPLEX_HEIGHT_IMAGINARY: &str = "0.0000002610401968271904579892363239208961136987073364114572308551191518591406534713386308202671231316320671575553797545468788348053202859995839467694832409411194930736";

fn assert_reference_values(digits: u64, base_imaginary: &str, references: &[(&str, &str, &str)]) {
    // Independent300-digit exp/log recurrences and triangular composition
    // at two orbit depths agree beyond126 component-relative digits.
    let reference_prec = cnum::digits_to_bits(180);
    let prec =
        cnum::checked_input_precision(digits, &[BASE, base_imaginary, "0.5", "0.25"]).unwrap();
    let base = cnum::parse_complex(BASE, base_imaginary, prec).unwrap();
    let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap() else {
        panic!("expected a negative-multiplier attracting boundary");
    };
    assert!(fp.lambda_abs < 1 && *fp.lambda.real() < 0);
    assert_eq!(fp.lambda.imag().is_zero(), base_imaginary == "0");
    let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
    for &(height_imaginary, re, im) in references {
        let height = cnum::parse_complex("0.5", height_imaginary, prec).unwrap();
        let actual = schroder::eval_schroder_at_digits(&state, &height, digits).unwrap();
        let expected = cnum::parse_complex(re, im, reference_prec).unwrap();
        for (actual, expected) in [
            (actual.real(), expected.real()),
            (actual.imag(), expected.imag()),
        ] {
            let error = Float::with_val_64(reference_prec, actual - expected).abs();
            assert!(error < cnum::epsilon(digits, reference_prec) * expected.clone().abs());
        }
        assert_eq!(
            cnum::format_complex(&actual, digits as usize),
            cnum::format_complex(&expected, digits as usize),
            "base={BASE}+{base_imaginary}i, height=0.5+{height_imaginary}i, digits={digits}",
        );
    }
}

fn assert_real_references(digits: u64) {
    assert_reference_values(
        digits,
        "0",
        &[
            ("0", REAL_HEIGHT_REAL, REAL_HEIGHT_IMAGINARY),
            ("0.25", COMPLEX_HEIGHT_REAL, COMPLEX_HEIGHT_IMAGINARY),
        ],
    );
}

fn assert_complex_references(digits: u64) {
    assert_reference_values(
        digits,
        "1e-14",
        &[
            (
                "0",
                "0.3678794157769882387951496017611623903414559461613295009887543756322903909347003020372212032355052144849672458024204746838382410773168355906228329753271358291117",
                "0.0000004456608585671471605101280693075758752176409204153453311321543491220127916207312630334333855450391264273628563899777127939695847888986452840967799587010044353281",
            ),
            (
                "0.25",
                "0.3678794262728258691799471402423521461713928961920603970905442644950975768580302318960208815771825452668640589014180804251727153564592838085530838927389154184178",
                "0.0000002614650206398378426653825988782434057475998374022886147188806945294439713663573599018401789920785552167736469351964510098787830619969858981381617651826672708383",
            ),
        ],
    );
}

#[test]
fn paired_real_base_20_digits() {
    assert_real_references(20);
}

#[test]
fn paired_real_base_50_digits() {
    assert_real_references(50);
}

#[test]
fn paired_real_base_100_digits() {
    assert_real_references(100);
}

#[test]
fn paired_complex_base_20_digits() {
    assert_complex_references(20);
}

#[test]
fn paired_complex_base_50_digits() {
    assert_complex_references(50);
}

#[test]
fn paired_complex_base_100_digits() {
    assert_complex_references(100);
}

fn assert_native_height(real: &str, imaginary: &str, expected_imaginary: &str) {
    let reference =
        cnum::parse_complex("1", expected_imaginary, cnum::digits_to_bits(180)).unwrap();
    let (expected_real, expected_imaginary) = cnum::format_complex(&reference, 50);
    let output = Command::new(env!("CARGO_BIN_EXE_tet"))
        .args(["--quiet", "50", "0.066", "0", real, imaginary])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    assert_eq!(
        output.stdout,
        format!("{expected_real}\n{expected_imaginary}\n").as_bytes()
    );
}

#[test]
fn negative_real_multiplier_preserves_native_imaginary_height() {
    assert_native_height(
        "0",
        "1e-1000000000000000000",
        "-1.303214795979021629785015636312762098904542023004103236850248059095236195813303300679525883986705734686470639889135249631836777619307017021979475841192043412979e-1000000000000000001",
    );
}

#[test]
fn negative_real_multiplier_preserves_native_real_height() {
    assert_native_height(
        "1e-1000000000000000000",
        "0",
        "1.227707188181169517968288048101230351922344797460149379411518881323973890110992394001271157767019069799561751317054771538504188894272970308916952545989873309442e-999999999999999996",
    );
}

#[test]
fn negative_near_neutral_cache_preserves_anchors_and_singularities() {
    let digits = 20;
    let prec = cnum::digits_to_bits(digits);
    let base = cnum::parse_complex(BASE, "0", prec).unwrap();
    let regions::Region::ShellThronBoundary(fp) = regions::classify(&base, prec).unwrap() else {
        panic!("expected a negative-multiplier attracting boundary");
    };
    let state = schroder::setup_schroder(&base, &fp, prec).unwrap();
    assert_eq!(
        schroder::eval_schroder_at_digits(&state, &cnum::zero(prec), digits).unwrap(),
        cnum::one(prec)
    );
    assert_eq!(
        schroder::eval_schroder_at_digits(&state, &cnum::one(prec), digits).unwrap(),
        base
    );
    let minus_one = Complex::with_val_64(prec, -1);
    assert!(cnum::is_zero(
        &schroder::eval_schroder_at_digits(&state, &minus_one, digits).unwrap()
    ));
    for height in [-2, -3] {
        let height = Complex::with_val_64(prec, height);
        assert!(schroder::eval_schroder_at_digits(&state, &height, digits)
            .unwrap_err()
            .contains("undefined"));
    }
}
