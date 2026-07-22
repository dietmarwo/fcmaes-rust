//! ESA GTOP space-mission benchmark functions.
//!
//! This is a safe Rust translation of `_fcmaescpp/gtop.cpp`. The public
//! functions keep the original decision-vector layouts and sanitize invalid
//! numerical results to the same large penalty used by the Python facade.

#![allow(clippy::excessive_precision)] // preserve the original GTOP constants verbatim

use std::f64::consts::PI;

type Vec3 = [f64; 3];

const PENALTY: f64 = 1.0e10;
const MU: [f64; 9] = [
    1.327_124_28e11,
    22_321.0,
    324_860.0,
    398_601.19,
    42_828.3,
    126.7e6,
    0.379_395_197_088_30e8,
    5.78e6,
    6.8e6,
];
const RPL: [f64; 6] = [2_440.0, 6_052.0, 6_378.0, 3_397.0, 71_492.0, 60_330.0];

#[inline]
fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn norm(v: &Vec3) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn distance(a: &Vec3, b: &Vec3) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

#[inline]
fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn add(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale(v: &Vec3, value: f64) -> Vec3 {
    [v[0] * value, v[1] * value, v[2] * value]
}

#[inline]
fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn unit(v: &Vec3) -> Vec3 {
    scale(v, 1.0 / norm(v))
}

fn sanitize(value: f64) -> f64 {
    if value.is_finite() { value } else { PENALTY }
}

fn bisect_root(mut lower: f64, mut upper: f64, function: impl Fn(f64) -> f64) -> f64 {
    let mut f_lower = function(lower);
    let f_upper = function(upper);
    if f_lower == 0.0 {
        return lower;
    }
    if f_upper == 0.0 {
        return upper;
    }
    if f_lower * f_upper > 0.0 {
        return 0.0;
    }
    for _ in 0..500 {
        let middle = 0.5 * (lower + upper);
        let f_middle = function(middle);
        if f_middle == 0.0 || upper - lower < 1.0e-15 {
            return middle;
        }
        if f_lower * f_middle <= 0.0 {
            upper = middle;
        } else {
            lower = middle;
            f_lower = f_middle;
        }
    }
    0.5 * (lower + upper)
}

fn mean_to_eccentric(mean_anomaly: f64, eccentricity: f64) -> f64 {
    if eccentricity < 1.0 {
        let mut eccentric = mean_anomaly + eccentricity * mean_anomaly.cos();
        for _ in 0..100 {
            let next = eccentric
                - (eccentric - eccentricity * eccentric.sin() - mean_anomaly)
                    / (1.0 - eccentricity * eccentric.cos());
            if (eccentric - next).abs() <= 1.0e-13 {
                return next;
            }
            eccentric = next;
        }
        eccentric
    } else {
        bisect_root(-0.5 * PI + 1.0e-8, 0.5 * PI - 1.0e-8, |value| {
            eccentricity * value.tan() - (0.5 * value + 0.25 * PI).tan().ln() - mean_anomaly
        })
    }
}

fn elements_to_state(elements: &[f64; 6], mu: f64) -> (Vec3, Vec3) {
    let [a, e, inclination, ascending_node, periapsis, anomaly] = *elements;
    let b = a * (1.0 - e * e).sqrt();
    let n = (mu / a.powi(3)).sqrt();
    let sin_anomaly = anomaly.sin();
    let cos_anomaly = anomaly.cos();
    let position_peri = [a * (cos_anomaly - e), b * sin_anomaly];
    let velocity_peri = [
        -(a * n * sin_anomaly) / (1.0 - e * cos_anomaly),
        (b * n * cos_anomaly) / (1.0 - e * cos_anomaly),
    ];
    let (sin_node, cos_node) = ascending_node.sin_cos();
    let (sin_peri, cos_peri) = periapsis.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    let rotation = [
        [
            cos_node * cos_peri - sin_node * sin_peri * cos_i,
            -cos_node * sin_peri - sin_node * cos_peri * cos_i,
        ],
        [
            sin_node * cos_peri + cos_node * sin_peri * cos_i,
            -sin_node * sin_peri + cos_node * cos_peri * cos_i,
        ],
        [sin_peri * sin_i, cos_peri * sin_i],
    ];
    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for row in 0..3 {
        position[row] = rotation[row][0] * position_peri[0] + rotation[row][1] * position_peri[1];
        velocity[row] = rotation[row][0] * velocity_peri[0] + rotation[row][1] * velocity_peri[1];
    }
    (position, velocity)
}

#[inline]
fn x_to_tof(x: f64, semiperimeter: f64, chord: f64, long_way: bool) -> f64 {
    let minimum_axis = semiperimeter / 2.0;
    let axis = minimum_axis / (1.0 - x * x);
    let (alpha, mut beta) = if x < 1.0 {
        (
            2.0 * x.acos(),
            2.0 * ((semiperimeter - chord) / (2.0 * axis)).sqrt().asin(),
        )
    } else {
        (
            2.0 * x.acosh(),
            2.0 * ((semiperimeter - chord) / (-2.0 * axis)).sqrt().asinh(),
        )
    };
    if long_way {
        beta = -beta;
    }
    if axis > 0.0 {
        axis * axis.sqrt() * ((alpha - alpha.sin()) - (beta - beta.sin()))
    } else {
        -axis * (-axis).sqrt() * ((alpha.sinh() - alpha) - (beta.sinh() - beta))
    }
}

/// Solve Lambert's boundary-value problem. This retains the original GTOP
/// secant iteration, including its single-precision `logf`/`expf` steps.
fn lambert(r1_input: &Vec3, r2_input: &Vec3, time: f64, mu: f64, long_way: bool) -> (Vec3, Vec3) {
    if time <= 0.0 {
        return ([f64::NAN; 3], [f64::NAN; 3]);
    }
    let radius = norm(r1_input);
    let velocity_scale = (mu / radius).sqrt();
    let time_scale = radius / velocity_scale;
    let nondim_time = time / time_scale;
    let r1 = scale(r1_input, 1.0 / radius);
    let r2 = scale(r2_input, 1.0 / radius);
    let r2_norm = norm(&r2);
    let mut theta = (dot(&r1, &r2) / r2_norm).acos();
    if long_way {
        theta = 2.0 * PI - theta;
    }
    let chord = (1.0 + r2_norm * (r2_norm - 2.0 * theta.cos())).sqrt();
    let semiperimeter = (1.0 + r2_norm + chord) / 2.0;
    let minimum_axis = semiperimeter / 2.0;
    let lambda = r2_norm.sqrt() * (theta / 2.0).cos() / semiperimeter;
    let mut x1 = 0.4767_f64.ln();
    let mut x2 = 1.5233_f64.ln();
    let mut y1 = x_to_tof(-0.5233, semiperimeter, chord, long_way).ln() - nondim_time.ln();
    let mut y2 = x_to_tof(0.5233, semiperimeter, chord, long_way).ln() - nondim_time.ln();
    let mut x_new = 0.0;
    for _ in 0..100 {
        if y1 == y2 {
            break;
        }
        x_new = (x1 * y2 - y1 * x2) / (y2 - y1);
        let tof = x_to_tof(
            (x_new as f32).exp() as f64 - 1.0,
            semiperimeter,
            chord,
            long_way,
        );
        let y_new = (tof as f32).ln() as f64 - (nondim_time as f32).ln() as f64;
        let previous_x2 = x2;
        x1 = x2;
        y1 = y2;
        x2 = x_new;
        y2 = y_new;
        if (previous_x2 - x_new).abs() <= 1.0e-11 {
            break;
        }
    }
    let x = (x_new as f32).exp() as f64 - 1.0;
    let axis = minimum_axis / (1.0 - x * x);
    let eta_squared = if x < 1.0 {
        let beta = 2.0 * ((semiperimeter - chord) / (2.0 * axis)).sqrt().asin();
        let alpha = 2.0 * x.acos();
        let signed_beta = if long_way { -beta } else { beta };
        let psi = (alpha - signed_beta) / 2.0;
        2.0 * axis * psi.sin().powi(2) / semiperimeter
    } else {
        let beta = 2.0 * ((chord - semiperimeter) / (2.0 * axis)).sqrt().asinh();
        let alpha = 2.0 * x.acosh();
        let signed_beta = if long_way { -beta } else { beta };
        let psi = (alpha - signed_beta) / 2.0;
        -2.0 * axis * psi.sinh().powi(2) / semiperimeter
    };
    let eta = eta_squared.sqrt();
    let parameter = r2_norm / (minimum_axis * eta_squared) * (theta / 2.0).sin().powi(2);
    let sigma1 = (2.0 * lambda * minimum_axis - (lambda + x * eta)) / (eta * minimum_axis.sqrt());
    let mut normal = unit(&cross(&r1, &r2));
    if long_way {
        normal = scale(&normal, -1.0);
    }
    let radial1 = sigma1;
    let tangent1 = parameter.sqrt();
    let v1 = add(&scale(&r1, radial1), &scale(&cross(&normal, &r1), tangent1));
    let tangent2 = tangent1 / r2_norm;
    let radial2 = -radial1 + (tangent1 - tangent2) / (theta / 2.0).tan();
    let r2_unit = unit(&r2);
    let v2 = add(
        &scale(&r2, radial2 / r2_norm),
        &scale(&cross(&normal, &r2_unit), tangent2),
    );
    (scale(&v1, velocity_scale), scale(&v2, velocity_scale))
}

fn planet_ephemerides(mjd2000: f64, planet: usize) -> (Vec3, Vec3) {
    let radians = PI / 180.0;
    let au = 149_597_870.66;
    let mut t = (mjd2000 + 36_525.0) / 36_525.0;
    let mut e = [0.0; 6];
    let mean_rate;
    match planet {
        1 => {
            e[0] = 0.387_098_60;
            e[1] = 0.205_614_210 + 0.000_020_460 * t - 0.000_000_030 * t * t;
            e[2] = 7.002_880_555_555_556 + 1.860_833_333_333_333e-3 * t
                - 1.833_333_333_333_333e-5 * t * t;
            e[3] = 47.145_944_444_444_44
                + 1.185_208_333_333_333_3 * t
                + 1.738_888_888_888_889e-4 * t * t;
            e[4] = 28.753_752_777_777_778
                + 0.370_280_555_555_555_56 * t
                + 1.208_333_333_333_333_3e-4 * t * t;
            mean_rate = 149_472.515_288_888_9 + 6.388_888_888_889e-6 * t;
            e[5] = 102.279_380_555_555_56 + mean_rate * t;
        }
        2 => {
            e[0] = 0.723_331_60;
            e[1] = 0.006_820_690 - 0.000_047_740 * t + 0.000_000_091 * t * t;
            e[2] = 3.393_630_555_555_555_6 + 1.005_833_333_333_333_4e-3 * t
                - 9.722_222_222_222_222e-7 * t * t;
            e[3] = 75.779_647_222_222_22 + 0.899_85 * t + 4.1e-4 * t * t;
            e[4] = 54.384_186_111_111_11 + 0.508_186_111_111_111_1 * t
                - 1.386_388_888_888_889e-3 * t * t;
            mean_rate = 58_517.803_875 + 1.286_055_555_555_555_5e-3 * t;
            e[5] = 212.603_219_444_444_44 + mean_rate * t;
        }
        3 => {
            e[0] = 1.000_000_230;
            e[1] = 0.016_751_040 - 0.000_041_800 * t - 0.000_000_126 * t * t;
            e[2] = 0.0;
            e[3] = 0.0;
            e[4] = 101.220_833_333_333_33
                + 1.719_175 * t
                + 4.527_777_777_777_778e-4 * t * t
                + 3.333_333_333_333_333_3e-6 * t.powi(3);
            mean_rate =
                35_999.049_75 - 1.502_777_777_777_777_8e-4 * t - 3.333_333_333_333_333_3e-6 * t * t;
            e[5] = 358.475_844_444_444_44 + mean_rate * t;
        }
        4 => {
            e[0] = 1.523_688_399;
            e[1] = 0.093_312_900 + 0.000_092_064 * t - 0.000_000_077 * t * t;
            e[2] = 1.850_333_333_333_333_3 - 6.75e-4 * t + 1.261_111_111_111_111_1e-5 * t * t;
            e[3] = 48.786_441_666_666_67 + 0.770_991_666_666_666_7 * t
                - 1.388_888_888_888_889e-6 * t * t
                - 5.333_333_333_333_333e-6 * t.powi(3);
            e[4] = 285.431_761_111_111_1
                + 1.069_766_666_666_666_8 * t
                + 1.312_5e-4 * t * t
                + 4.138_888_888_888_889e-6 * t.powi(3);
            mean_rate =
                19_139.858_5 + 1.808_055_555_555_555_5e-4 * t + 1.194_444_444_444_444_5e-6 * t * t;
            e[5] = 319.529_425 + mean_rate * t;
        }
        5 => {
            e[0] = 5.202_561;
            e[1] = 0.048_334_750 + 0.000_164_180 * t
                - 0.000_000_467_60 * t * t
                - 0.000_000_001_70 * t.powi(3);
            e[2] = 1.308_736_111_111_111 - 5.696_111_111_111_111e-3 * t
                + 3.888_888_888_888_889e-6 * t * t;
            e[3] = 99.443_386_111_111_11 + 1.010_530 * t + 3.522_222_222_222_222e-4 * t * t
                - 8.511_111_111_111_111e-6 * t.powi(3);
            e[4] = 273.277_541_666_666_67
                + 0.599_431_666_666_666_7 * t
                + 7.040_5e-4 * t * t
                + 5.077_777_777_777_778e-6 * t.powi(3);
            mean_rate = 3_034.692_023_888_889 - 7.215_888_888_888_889e-4 * t
                + 1.784_444_444_444_444_4e-6 * t * t;
            e[5] = 225.328_327_777_777_78 + mean_rate * t;
        }
        6 => {
            e[0] = 9.554_747;
            e[1] = 0.055_892_320 - 0.000_345_50 * t - 0.000_000_728 * t * t
                + 0.000_000_000_740 * t.powi(3);
            e[2] = 2.492_519_444_444_444_4
                - 3.918_888_888_888_889e-3 * t
                - 1.548_888_888_888_888_9e-5 * t * t
                + 4.444_444_444_444_444_4e-8 * t.powi(3);
            e[3] = 112.790_388_888_888_89 + 0.873_195_138_888_888_9 * t
                - 1.521_805_555_555_555_6e-4 * t * t
                - 5.305_555_555_555_556e-6 * t.powi(3);
            e[4] = 338.307_772_222_222_2
                + 1.085_220_694_444_444_4 * t
                + 9.785_416_666_666_667e-4 * t * t
                + 9.916_666_666_666_667e-6 * t.powi(3);
            mean_rate = 1_221.551_467_777_777_8
                - 5.018_194_444_444_444e-4 * t
                - 5.194_444_444_445e-6 * t * t;
            e[5] = 175.466_216_666_666_67 + mean_rate * t;
        }
        7 => {
            e[0] = 19.218_140;
            e[1] = 0.046_344_40 - 0.000_026_580 * t + 0.000_000_077 * t * t;
            e[2] = 0.772_463_888_888_888_9 + 6.252_777_777_777_778e-4 * t + 3.95e-5 * t * t;
            e[3] = 73.477_097_222_222_22
                + 0.498_667_777_777_777_8 * t
                + 1.311_666_666_666_666_7e-3 * t * t;
            e[4] = 98.071_552_777_777_78 + 0.985_765 * t
                - 1.074_472_222_222_222_2e-3 * t * t
                - 6.055_555_555_555_556e-7 * t.powi(3);
            mean_rate = 428.379_113_055_555_56
                + 7.884_444_444_444_444e-5 * t
                + 1.111_111_111_111_111e-9 * t * t;
            e[5] = 72.648_819_444_444_44 + mean_rate * t;
        }
        8 => {
            e[0] = 30.109_570;
            e[1] = 0.008_997_040 + 0.000_006_330 * t - 0.000_000_002 * t * t;
            e[2] = 1.779_241_666_666_666_7
                - 9.543_611_111_111_111e-3 * t
                - 9.111_111_111_111_111e-6 * t * t;
            e[3] = 130.681_358_333_333_33 + 1.098_935 * t + 2.498_666_666_666_667e-4 * t * t
                - 4.717_777_777_777_778e-6 * t.powi(3);
            e[4] = 276.045_966_666_666_67
                + 0.325_639_444_444_444_45 * t
                + 1.409_5e-4 * t * t
                + 4.113_333_333_333_333e-6 * t.powi(3);
            mean_rate = 218.461_339_722_222_22 - 7.033_333_333_333_333e-5 * t;
            e[5] = 37.730_669_444_444_44 + mean_rate * t;
        }
        9 => {
            t = mjd2000 / 36_525.0;
            e[0] = 39.340_419_612_525_20 + 4.333_051_381_207_26 * t
                - 22.937_499_324_037_33 * t.powi(2)
                + 48.763_367_207_918_73 * t.powi(3)
                - 45.524_948_624_623_79 * t.powi(4)
                + 15.551_349_517_833_84 * t.powi(5);
            e[1] = 0.246_173_653_965_17 + 0.091_980_017_421_90 * t
                - 0.572_622_889_914_47 * t.powi(2)
                + 1.391_630_228_810_98 * t.powi(3)
                - 1.469_484_515_876_83 * t.powi(4)
                + 0.561_641_587_216_20 * t.powi(5);
            e[2] = 17.166_900_037_847_02 - 0.497_702_487_904_79 * t
                + 2.737_519_018_908_29 * t.powi(2)
                - 6.269_736_951_975_47 * t.powi(3)
                + 6.362_769_273_974_30 * t.powi(4)
                - 2.370_069_116_730_31 * t.powi(5);
            e[3] = 110.222_019_291_707 + 1.551_579_150_048 * t - 9.701_771_291_171 * t.powi(2)
                + 25.730_756_810_615 * t.powi(3)
                - 30.140_401_383_522 * t.powi(4)
                + 12.796_598_193_159 * t.powi(5);
            e[4] = 113.368_933_916_592 + 9.436_835_192_183 * t - 35.762_300_003_726 * t.powi(2)
                + 48.966_118_351_549 * t.powi(3)
                - 19.384_576_636_609 * t.powi(4)
                - 3.362_714_022_614 * t.powi(5);
            e[5] = 15.170_086_316_346_65 + 137.023_166_578_486 * t + 28.362_805_871_736 * t.powi(2)
                - 29.677_368_415_909 * t.powi(3)
                - 3.585_159_909_117 * t.powi(4)
                + 13.406_844_652_829 * t.powi(5);
        }
        _ => return ([f64::NAN; 3], [f64::NAN; 3]),
    }
    e[0] *= au;
    for value in &mut e[2..] {
        *value *= radians;
    }
    e[5] %= 2.0 * PI;
    e[5] = mean_to_eccentric(e[5], e[1]);
    elements_to_state(&e, MU[0])
}

#[derive(Clone, Debug, Default)]
struct CustomObject {
    keplerian: [f64; 6],
    epoch: f64,
    mu: f64,
}

fn custom_ephemerides(julian_date: f64, object: &CustomObject) -> (Vec3, Vec3) {
    let au = 149_597_870.66;
    let mut elements = [0.0; 6];
    elements[0] = object.keplerian[0] * au;
    elements[1] = object.keplerian[1];
    elements[2] = object.keplerian[2].to_radians();
    elements[3] = object.keplerian[3].to_radians();
    elements[4] = object.keplerian[4].to_radians();
    let elapsed = (julian_date - (object.epoch + 2_400_000.5)) * 86_400.0;
    let mean_motion = (MU[0] / elements[0].powi(3)).sqrt();
    let mean = (object.keplerian[5].to_radians() + mean_motion * elapsed) % (2.0 * PI);
    elements[5] = mean_to_eccentric(mean, elements[1]);
    elements_to_state(&elements, MU[0])
}

fn powered_swingby(incoming: f64, outgoing: f64, alpha: f64) -> (f64, f64) {
    let incoming_axis = 1.0 / incoming.powi(2);
    let outgoing_axis = 1.0 / outgoing.powi(2);
    let mut periapsis = 1.0;
    for _ in 0..30 {
        let function = (incoming_axis / (incoming_axis + periapsis)).asin()
            + (outgoing_axis / (outgoing_axis + periapsis)).asin()
            - alpha;
        let derivative = -incoming_axis
            / ((periapsis + 2.0 * incoming_axis) * periapsis).sqrt()
            / (incoming_axis + periapsis)
            - outgoing_axis
                / ((periapsis + 2.0 * outgoing_axis) * periapsis).sqrt()
                / (outgoing_axis + periapsis);
        let next = periapsis - function / derivative;
        if next > 0.0 {
            if (next - periapsis).abs() <= 1.0e-8 {
                periapsis = next;
                break;
            }
            periapsis = next;
        } else {
            periapsis /= 2.0;
        }
    }
    let delta_v = ((outgoing.powi(2) + 2.0 / periapsis).sqrt()
        - (incoming.powi(2) + 2.0 / periapsis).sqrt())
    .abs();
    (delta_v, periapsis)
}

fn state_to_elements(position: &Vec3, velocity: &Vec3, mu: f64) -> [f64; 6] {
    let angular_momentum = cross(position, velocity);
    let parameter = dot(&angular_momentum, &angular_momentum) / mu;
    let node = unit(&cross(&[0.0, 0.0, 1.0], &angular_momentum));
    let radius = norm(position);
    let eccentricity_vector = sub(
        &scale(&cross(velocity, &angular_momentum), 1.0 / mu),
        &scale(position, 1.0 / radius),
    );
    let eccentricity = norm(&eccentricity_vector);
    let mut elements = [0.0; 6];
    elements[0] = parameter / (1.0 - eccentricity.powi(2));
    elements[1] = eccentricity;
    elements[2] = (angular_momentum[2] / norm(&angular_momentum)).acos();
    elements[4] = (dot(&node, &eccentricity_vector) / eccentricity).acos();
    if eccentricity_vector[2] < 0.0 {
        elements[4] = 2.0 * PI - elements[4];
    }
    elements[3] = node[0].acos();
    if node[1] < 0.0 {
        elements[3] = 2.0 * PI - elements[3];
    }
    let mut true_anomaly = (dot(&eccentricity_vector, position) / eccentricity / radius).acos();
    if dot(position, velocity) < 0.0 {
        true_anomaly = 2.0 * PI - true_anomaly;
    }
    elements[5] = if eccentricity < 1.0 {
        2.0 * (((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt() * (true_anomaly / 2.0).tan())
            .atan()
    } else {
        2.0 * (((eccentricity - 1.0) / (eccentricity + 1.0)).sqrt() * (true_anomaly / 2.0).tan())
            .atan()
    };
    elements
}

fn full_elements_to_state(elements: &[f64; 6], mu: f64) -> (Vec3, Vec3) {
    let [a, e, inclination, node, periapsis, anomaly] = *elements;
    let (x, y, vx, vy) = if e < 1.0 {
        let b = a * (1.0 - e * e).sqrt();
        let n = (mu / a.powi(3)).sqrt();
        (
            a * (anomaly.cos() - e),
            b * anomaly.sin(),
            -(a * n * anomaly.sin()) / (1.0 - e * anomaly.cos()),
            (b * n * anomaly.cos()) / (1.0 - e * anomaly.cos()),
        )
    } else {
        let b = -a * (e * e - 1.0).sqrt();
        let n = (-mu / a.powi(3)).sqrt();
        let tangent = anomaly.tan();
        let half_tangent = (0.5 * anomaly + 0.25 * PI).tan();
        let derivative =
            e * (1.0 + tangent * tangent) - (0.5 + 0.5 * half_tangent.powi(2)) / half_tangent;
        (
            a / anomaly.cos() - a * e,
            b * tangent,
            a * tangent / anomaly.cos() * n / derivative,
            b / anomaly.cos().powi(2) * n / derivative,
        )
    };
    let (sin_node, cos_node) = node.sin_cos();
    let (sin_peri, cos_peri) = periapsis.sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    let rotation = [
        [
            cos_node * cos_peri - sin_node * sin_peri * cos_i,
            -cos_node * sin_peri - sin_node * cos_peri * cos_i,
            sin_node * sin_i,
        ],
        [
            sin_node * cos_peri + cos_node * sin_peri * cos_i,
            -sin_node * sin_peri + cos_node * cos_peri * cos_i,
            -cos_node * sin_i,
        ],
        [sin_peri * sin_i, cos_peri * sin_i, cos_i],
    ];
    let mut position = [0.0; 3];
    let mut velocity = [0.0; 3];
    for row in 0..3 {
        position[row] = rotation[row][0] * x + rotation[row][1] * y;
        velocity[row] = rotation[row][0] * vx + rotation[row][1] * vy;
    }
    (position, velocity)
}

fn propagate_kepler(
    position_input: &Vec3,
    velocity_input: &Vec3,
    time: f64,
    mu: f64,
) -> (Vec3, Vec3) {
    let mut position = *position_input;
    let mut velocity = *velocity_input;
    let direction = unit(&cross(&position, &velocity));
    let rotate = (direction[2].abs() - 1.0).abs() < 1.0e-3;
    if rotate {
        position = [position[0], position[2], -position[1]];
        velocity = [velocity[0], velocity[2], -velocity[1]];
    }
    let mut elements = state_to_elements(&position, &velocity, mu);
    let mean = if elements[1] < 1.0 {
        elements[5] - elements[1] * elements[5].sin() + (mu / elements[0].powi(3)).sqrt() * time
    } else {
        elements[1] * elements[5].tan() - (0.5 * elements[5] + 0.25 * PI).tan().ln()
            + (mu / (-elements[0]).powi(3)).sqrt() * time
    };
    elements[5] = mean_to_eccentric(mean, elements[1]);
    let (mut result_position, mut result_velocity) = full_elements_to_state(&elements, mu);
    if rotate {
        result_position = [result_position[0], -result_position[2], result_position[1]];
        result_velocity = [result_velocity[0], -result_velocity[2], result_velocity[1]];
    }
    (result_position, result_velocity)
}

fn time_to_distance(position: &Vec3, velocity: &Vec3, target: f64) -> f64 {
    let radius = norm(position);
    if radius >= target {
        return 12.0;
    }
    let radial_velocity = dot(position, velocity);
    let elements = state_to_elements(position, velocity, 1.0);
    let axis = elements[0];
    let eccentricity = elements[1];
    let mut initial_anomaly = elements[5];
    let parameter = axis * (1.0 - eccentricity * eccentricity);
    if eccentricity < 1.0 {
        let apoapsis = axis * (1.0 + eccentricity);
        if target > apoapsis {
            return -1.0;
        }
        let true_anomaly = ((parameter / target - 1.0) / eccentricity).acos();
        let target_anomaly = 2.0
            * (((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt() * (true_anomaly / 2.0).tan())
                .atan();
        if radial_velocity > 0.0 {
            axis.powi(3).sqrt()
                * (target_anomaly - eccentricity * target_anomaly.sin() - initial_anomaly
                    + eccentricity * initial_anomaly.sin())
        } else {
            initial_anomaly = -initial_anomaly;
            axis.powi(3).sqrt()
                * (target_anomaly - eccentricity * target_anomaly.sin() + initial_anomaly
                    - eccentricity * initial_anomaly.sin())
        }
    } else {
        let true_anomaly = ((parameter / target - 1.0) / eccentricity).acos();
        let target_anomaly = 2.0
            * (((eccentricity - 1.0) / (eccentricity + 1.0)).sqrt() * (true_anomaly / 2.0).tan())
                .atan();
        let equation =
            |value: f64| eccentricity * value.tan() - (value / 2.0 + PI / 4.0).tan().ln();
        if radial_velocity > 0.0 {
            (-axis).powi(3).sqrt() * (equation(target_anomaly) - equation(initial_anomaly))
        } else {
            initial_anomaly = -initial_anomaly;
            (-axis).powi(3).sqrt() * (equation(target_anomaly) + equation(initial_anomaly))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MissionType {
    OrbitInsertion,
    TotalDvOrbitInsertion,
    Rendezvous,
    TotalDvRendezvous,
    AsteroidImpact,
    TimeToAus,
}

#[derive(Clone, Debug)]
struct MgaProblem {
    mission: MissionType,
    sequence: Vec<usize>,
    reverse: Vec<bool>,
    eccentricity: f64,
    periapsis: f64,
    asteroid: CustomObject,
    specific_impulse: f64,
    mass: f64,
    launch_delta_v: f64,
}

impl Default for MgaProblem {
    fn default() -> Self {
        Self {
            mission: MissionType::TotalDvOrbitInsertion,
            sequence: Vec::new(),
            reverse: Vec::new(),
            eccentricity: 0.0,
            periapsis: 0.0,
            asteroid: CustomObject::default(),
            specific_impulse: 0.0,
            mass: 0.0,
            launch_delta_v: 0.0,
        }
    }
}

/// Multiple-gravity-assist model used by Cassini 1 and GTOC1.
fn mga(times: &[f64], problem: &MgaProblem) -> (f64, Vec<f64>, f64) {
    let n = problem.sequence.len();
    if n < 2 || times.len() < n || problem.reverse.len() < n {
        return (f64::NAN, Vec::new(), f64::NAN);
    }
    const MINIMUM_PERIAPSIS: [f64; 9] = [
        0.0, 0.0, 6_351.8, 6_778.1, 6_000.0, 600_000.0, 70_000.0, 0.0, 0.0,
    ];
    const PENALTY_COEFFICIENT: [f64; 9] = [0.0, 0.0, 0.01, 0.01, 0.01, 0.001, 0.01, 0.0, 0.0];
    // The older MGA routine uses a rounded Saturn constant; MGA-DSM below
    // uses the later, more precise value.
    const MGA_MU: [f64; 9] = [
        1.327_124_28e11,
        22_321.0,
        324_860.0,
        398_601.19,
        42_828.3,
        126.7e6,
        37.9e6,
        5.78e6,
        6.8e6,
    ];

    let mut positions = vec![[0.0; 3]; n];
    let mut velocities = vec![[0.0; 3]; n];
    let mut epoch = 0.0;
    for i in 0..n {
        epoch += times[i];
        (positions[i], velocities[i]) = if problem.sequence[i] < 10 {
            planet_ephemerides(epoch, problem.sequence[i])
        } else {
            custom_ephemerides(epoch + 2_451_544.5, &problem.asteroid)
        };
    }

    let long_way = (cross(&positions[0], &positions[1])[2] > 0.0) == problem.reverse[0];
    let mut previous_transfer = lambert(
        &positions[0],
        &positions[1],
        times[1] * 86_400.0,
        MGA_MU[0],
        long_way,
    );
    let mut delta_v = vec![0.0; n];
    delta_v[0] = distance(&previous_transfer.0, &velocities[0]);
    let mut periapses = vec![0.0; n - 2];

    for i in 1..=n - 2 {
        let long_way = (cross(&positions[i], &positions[i + 1])[2] > 0.0) == problem.reverse[i];
        let next_transfer = lambert(
            &positions[i],
            &positions[i + 1],
            times[i + 1] * 86_400.0,
            MGA_MU[0],
            long_way,
        );
        let incoming = distance(&previous_transfer.1, &velocities[i]);
        let outgoing = distance(&next_transfer.0, &velocities[i]);
        let incoming_relative = sub(&previous_transfer.1, &velocities[i]);
        let outgoing_relative = sub(&next_transfer.0, &velocities[i]);
        let angle = (dot(&incoming_relative, &outgoing_relative) / (incoming * outgoing)).acos();
        let (powered_delta_v, nondimensional_periapsis) =
            powered_swingby(incoming, outgoing, angle);
        delta_v[i] = powered_delta_v;
        periapses[i - 1] = nondimensional_periapsis * MGA_MU[problem.sequence[i]];
        previous_transfer = next_transfer;
    }

    let arrival_relative = distance(&velocities[n - 1], &previous_transfer.1);
    let arrival_delta_v = match problem.mission {
        MissionType::TotalDvOrbitInsertion => {
            let mu = MGA_MU[problem.sequence[n - 1]];
            let hyperbolic = (arrival_relative.powi(2) + 2.0 * mu / problem.periapsis).sqrt();
            let target = (2.0 * mu / problem.periapsis
                - mu / problem.periapsis * (1.0 - problem.eccentricity))
                .sqrt();
            (hyperbolic - target).abs()
        }
        MissionType::AsteroidImpact => arrival_relative,
        _ => 0.0,
    };

    let mut total_delta_v: f64 = delta_v[1..n - 1].iter().sum();
    if problem.mission == MissionType::TotalDvOrbitInsertion {
        total_delta_v += arrival_delta_v;
    }
    for (i, &periapsis) in periapses.iter().enumerate() {
        let body = problem.sequence[i + 1];
        if periapsis < MINIMUM_PERIAPSIS[body] {
            total_delta_v +=
                PENALTY_COEFFICIENT[body] * (periapsis - MINIMUM_PERIAPSIS[body]).abs();
        }
    }
    if delta_v[0] > problem.launch_delta_v {
        total_delta_v += delta_v[0] - problem.launch_delta_v;
    }

    let objective = match problem.mission {
        MissionType::TotalDvOrbitInsertion => total_delta_v,
        MissionType::AsteroidImpact => {
            let gravity = 9.806_65 / 1_000.0;
            let final_mass =
                problem.mass * (-total_delta_v / (problem.specific_impulse * gravity)).exp();
            let relative = sub(&velocities[n - 1], &previous_transfer.1);
            2_000_000.0 - final_mass * dot(&relative, &velocities[n - 1]).abs()
        }
        _ => f64::NAN,
    };
    (objective, periapses, delta_v[0])
}

#[derive(Clone, Debug)]
struct DsmProblem {
    mission: MissionType,
    sequence: Vec<usize>,
    eccentricity: f64,
    periapsis: f64,
    asteroid: CustomObject,
    au_distance: f64,
    total_delta_v_limit: f64,
    onboard_delta_v_limit: f64,
}

impl DsmProblem {
    fn new(mission: MissionType, sequence: &[usize]) -> Self {
        Self {
            mission,
            sequence: sequence.to_vec(),
            eccentricity: 0.0,
            periapsis: 0.0,
            asteroid: CustomObject::default(),
            au_distance: 0.0,
            total_delta_v_limit: 0.0,
            onboard_delta_v_limit: 0.0,
        }
    }

    fn state(&self, epoch: f64, index: usize) -> (Vec3, Vec3) {
        if self.sequence[index] < 10 {
            planet_ephemerides(epoch, self.sequence[index])
        } else {
            custom_ephemerides(epoch + 2_451_544.5, &self.asteroid)
        }
    }

    fn body_mu(&self, index: usize) -> f64 {
        if self.sequence[index] < 10 {
            MU[self.sequence[index]]
        } else {
            self.asteroid.mu
        }
    }
}

fn dsm_first_block(
    decision: &[f64],
    problem: &DsmProblem,
    positions: &[Vec3],
    velocities: &[Vec3],
    delta_v: &mut [f64],
) -> Vec3 {
    let n = problem.sequence.len();
    let tof = &decision[4..];
    let alpha = &decision[n + 3..];
    let normal = unit(&cross(&positions[0], &velocities[0]));
    let tangent = unit(&velocities[0]);
    let transverse = cross(&normal, &tangent);
    let theta = 2.0 * PI * decision[2];
    let phi = (2.0 * decision[3] - 1.0).acos() - PI / 2.0;
    let vinf = [
        decision[1]
            * (theta.cos() * phi.cos() * tangent[0]
                + theta.sin() * phi.cos() * transverse[0]
                + phi.sin() * normal[0]),
        decision[1]
            * (theta.cos() * phi.cos() * tangent[1]
                + theta.sin() * phi.cos() * transverse[1]
                + phi.sin() * normal[1]),
        decision[1]
            * (theta.cos() * phi.cos() * tangent[2]
                + theta.sin() * phi.cos() * transverse[2]
                + phi.sin() * normal[2]),
    ];
    let outgoing = add(&velocities[0], &vinf);
    let (dsm_position, dsm_incoming) = propagate_kepler(
        &positions[0],
        &outgoing,
        alpha[0] * tof[0] * 86_400.0,
        MU[0],
    );
    let long_way = cross(&dsm_position, &positions[1])[2] <= 0.0;
    let (dsm_outgoing, next_incoming) = lambert(
        &dsm_position,
        &positions[1],
        tof[0] * (1.0 - alpha[0]) * 86_400.0,
        MU[0],
        long_way,
    );
    delta_v[0] = distance(&dsm_outgoing, &dsm_incoming);
    next_incoming
}

fn dsm_intermediate_block(
    decision: &[f64],
    problem: &DsmProblem,
    positions: &[Vec3],
    velocities: &[Vec3],
    index: usize,
    planet_incoming: &Vec3,
    delta_v: &mut [f64],
) -> Vec3 {
    let n = problem.sequence.len();
    let tof = &decision[4..];
    let alpha = &decision[n + 3..];
    let nondimensional_periapsis = &decision[2 * n + 2..];
    let gamma = &decision[3 * n..];
    let relative_incoming = sub(planet_incoming, &velocities[index + 1]);
    let relative_speed_squared = dot(&relative_incoming, &relative_incoming);
    let eccentricity = 1.0
        + nondimensional_periapsis[index]
            * RPL[problem.sequence[index + 1] - 1]
            * relative_speed_squared
            / problem.body_mu(index + 1);
    let rotation = 2.0 * (1.0 / eccentricity).asin();
    let x_axis = unit(&relative_incoming);
    let velocity_axis = unit(&velocities[index + 1]);
    let y_axis = unit(&cross(&x_axis, &velocity_axis));
    let z_axis = cross(&x_axis, &y_axis);
    let relative_speed = norm(&relative_incoming);
    let rotated = [
        rotation.cos() * x_axis[0]
            + gamma[index].cos() * rotation.sin() * y_axis[0]
            + gamma[index].sin() * rotation.sin() * z_axis[0],
        rotation.cos() * x_axis[1]
            + gamma[index].cos() * rotation.sin() * y_axis[1]
            + gamma[index].sin() * rotation.sin() * z_axis[1],
        rotation.cos() * x_axis[2]
            + gamma[index].cos() * rotation.sin() * y_axis[2]
            + gamma[index].sin() * rotation.sin() * z_axis[2],
    ];
    let planet_outgoing = add(&velocities[index + 1], &scale(&rotated, relative_speed));
    let (dsm_position, dsm_incoming) = propagate_kepler(
        &positions[index + 1],
        &planet_outgoing,
        alpha[index + 1] * tof[index + 1] * 86_400.0,
        MU[0],
    );
    let long_way = cross(&dsm_position, &positions[index + 2])[2] <= 0.0;
    let (dsm_outgoing, next_incoming) = lambert(
        &dsm_position,
        &positions[index + 2],
        tof[index + 1] * (1.0 - alpha[index + 1]) * 86_400.0,
        MU[0],
        long_way,
    );
    delta_v[index + 1] = distance(&dsm_outgoing, &dsm_incoming);
    next_incoming
}

fn dsm_final_delta_v(problem: &DsmProblem, target_velocity: &Vec3, incoming: &Vec3) -> f64 {
    let relative = distance(target_velocity, incoming);
    match problem.mission {
        MissionType::OrbitInsertion | MissionType::TotalDvOrbitInsertion => {
            let target = problem.sequence.len() - 1;
            let mu = MU[problem.sequence[target]];
            let hyperbolic = (relative.powi(2) + 2.0 * mu / problem.periapsis).sqrt();
            let target_speed = (2.0 * mu / problem.periapsis
                - mu / problem.periapsis * (1.0 - problem.eccentricity))
                .sqrt();
            (hyperbolic - target_speed).abs()
        }
        MissionType::Rendezvous | MissionType::TotalDvRendezvous => relative,
        _ => 0.0,
    }
}

/// Multiple-gravity-assist with one deep-space maneuver per leg.
fn mga_dsm(decision: &[f64], problem: &DsmProblem) -> (f64, Vec<f64>) {
    let n = problem.sequence.len();
    if n < 2 || decision.len() < 4 * n - 2 {
        return (f64::NAN, Vec::new());
    }
    let mut positions = vec![[0.0; 3]; n];
    let mut velocities = vec![[0.0; 3]; n];
    let mut epoch = decision[0];
    for i in 0..n {
        (positions[i], velocities[i]) = problem.state(epoch, i);
        epoch += decision[4 + i];
    }
    let mut delta_v = vec![0.0; n + 2];
    let mut incoming = dsm_first_block(decision, problem, &positions, &velocities, &mut delta_v);
    for index in 0..n - 2 {
        incoming = dsm_intermediate_block(
            decision,
            problem,
            &positions,
            &velocities,
            index,
            &incoming,
            &mut delta_v,
        );
    }
    delta_v[n - 1] = dsm_final_delta_v(problem, &velocities[n - 1], &incoming);
    let total: f64 = delta_v[..n].iter().sum();
    for index in (1..=n).rev() {
        delta_v[index] = delta_v[index - 1];
    }
    delta_v[0] = decision[1];

    let objective = match problem.mission {
        MissionType::OrbitInsertion | MissionType::Rendezvous => total,
        MissionType::TotalDvOrbitInsertion | MissionType::TotalDvRendezvous => total + decision[1],
        MissionType::TimeToAus => {
            let nondimensional_periapsis = &decision[2 * n + 2..];
            let gamma = &decision[3 * n..];
            let relative_incoming = sub(&incoming, &velocities[n - 1]);
            let relative_speed_squared = dot(&relative_incoming, &relative_incoming);
            let eccentricity = 1.0
                + nondimensional_periapsis[n - 2]
                    * RPL[problem.sequence[n - 1] - 1]
                    * relative_speed_squared
                    / problem.body_mu(n - 1);
            let rotation = 2.0 * (1.0 / eccentricity).asin();
            let x_axis = unit(&relative_incoming);
            let y_axis = unit(&cross(&x_axis, &unit(&velocities[n - 1])));
            let z_axis = cross(&x_axis, &y_axis);
            let rotated = [
                rotation.cos() * x_axis[0]
                    + gamma[n - 2].cos() * rotation.sin() * y_axis[0]
                    + gamma[n - 2].sin() * rotation.sin() * z_axis[0],
                rotation.cos() * x_axis[1]
                    + gamma[n - 2].cos() * rotation.sin() * y_axis[1]
                    + gamma[n - 2].sin() * rotation.sin() * z_axis[1],
                rotation.cos() * x_axis[2]
                    + gamma[n - 2].cos() * rotation.sin() * y_axis[2]
                    + gamma[n - 2].sin() * rotation.sin() * z_axis[2],
            ];
            let outgoing = add(
                &velocities[n - 1],
                &scale(&rotated, norm(&relative_incoming)),
            );
            let au = 149_597_870.66;
            let velocity_scale = (MU[0] / au).sqrt();
            let time_scale = au / velocity_scale;
            let travel_time = time_to_distance(
                &scale(&positions[n - 1], 1.0 / au),
                &scale(&outgoing, 1.0 / velocity_scale),
                problem.au_distance,
            );
            if travel_time == -1.0 {
                100_000.0
            } else {
                let flight_days: f64 = decision[4..4 + n - 1].iter().sum();
                (travel_time * time_scale / 86_400.0 + flight_days) / 365.25
            }
        }
        MissionType::AsteroidImpact => f64::NAN,
    };
    (objective, delta_v)
}

/// GTOC1 asteroid-impact benchmark.
pub fn gtoc1(x: &[f64]) -> f64 {
    if x.len() < 8 {
        return PENALTY;
    }
    let problem = MgaProblem {
        mission: MissionType::AsteroidImpact,
        sequence: vec![3, 2, 3, 2, 3, 5, 6, 10],
        reverse: vec![false, false, false, false, false, false, true, false],
        asteroid: CustomObject {
            keplerian: [
                2.589_726_1,
                0.273_462_5,
                6.407_34,
                128.347_11,
                264.786_91,
                320.479_555,
            ],
            epoch: 53_600.0,
            mu: 0.0,
        },
        specific_impulse: 2_500.0,
        mass: 1_500.0,
        launch_delta_v: 2.5,
        ..Default::default()
    };
    sanitize(mga(x, &problem).0)
}

fn cassini1_details(x: &[f64], sequence: &[usize]) -> (f64, Vec<f64>, f64) {
    if x.len() < 6 {
        return (PENALTY, Vec::new(), PENALTY);
    }
    let problem = MgaProblem {
        mission: MissionType::TotalDvOrbitInsertion,
        sequence: sequence.to_vec(),
        reverse: vec![false; 6],
        eccentricity: 0.98,
        periapsis: 108_950.0,
        launch_delta_v: 0.0,
        ..Default::default()
    };
    let (value, periapses, launch) = mga(x, &problem);
    (sanitize(value), periapses, sanitize(launch))
}

/// Cassini 1 benchmark.
pub fn cassini1(x: &[f64]) -> f64 {
    cassini1_details(x, &[3, 2, 2, 3, 5, 6]).0
}

/// Mixed-integer Cassini 1 benchmark, returning objective and launch delta-v.
pub fn cassini1_minlp(x: &[f64]) -> (f64, f64) {
    if x.len() < 10 {
        return (PENALTY, PENALTY);
    }
    let sequence = [
        3,
        x[6] as usize,
        x[7] as usize,
        x[8] as usize,
        x[9] as usize,
        6,
    ];
    let (value, _, launch) = cassini1_details(x, &sequence);
    (value, launch)
}

/// Reduced Messenger benchmark.
pub fn messenger(x: &[f64]) -> f64 {
    if x.len() < 18 {
        return PENALTY;
    }
    let problem = DsmProblem::new(MissionType::TotalDvRendezvous, &[3, 3, 2, 2, 1]);
    sanitize(mga_dsm(x, &problem).0)
}

/// Full Messenger benchmark.
pub fn messenger_full(x: &[f64]) -> f64 {
    if x.len() < 26 {
        return PENALTY;
    }
    let mut problem = DsmProblem::new(MissionType::OrbitInsertion, &[3, 2, 2, 1, 1, 1, 1]);
    problem.eccentricity = 0.704;
    problem.periapsis = 2_640.0;
    sanitize(mga_dsm(x, &problem).0)
}

/// Cassini 2 benchmark.
pub fn cassini2(x: &[f64]) -> f64 {
    if x.len() < 22 {
        return PENALTY;
    }
    let problem = DsmProblem::new(MissionType::TotalDvRendezvous, &[3, 2, 2, 3, 5, 6]);
    sanitize(mga_dsm(x, &problem).0)
}

/// Mixed-integer Cassini 2 benchmark.
pub fn cassini2_minlp(x: &[f64]) -> f64 {
    if x.len() < 26 {
        return PENALTY;
    }
    let sequence = [
        3,
        x[22] as usize,
        x[23] as usize,
        x[24] as usize,
        x[25] as usize,
        6,
    ];
    if sequence.iter().any(|&body| !(1..=6).contains(&body)) {
        return PENALTY;
    }
    let problem = DsmProblem::new(MissionType::TotalDvRendezvous, &sequence);
    sanitize(mga_dsm(x, &problem).0)
}

/// Rosetta rendezvous benchmark.
pub fn rosetta(x: &[f64]) -> f64 {
    if x.len() < 22 {
        return PENALTY;
    }
    let mut problem = DsmProblem::new(MissionType::Rendezvous, &[3, 3, 4, 3, 3, 10]);
    problem.asteroid = CustomObject {
        keplerian: [
            3.502_949_728_362_75,
            0.631_935_6,
            7.127_23,
            50.923_02,
            11.367_88,
            0.0,
        ],
        epoch: 52_504.237_540_000_12,
        mu: 0.0,
    };
    sanitize(mga_dsm(x, &problem).0)
}

/// SAGAS time-to-50-AU benchmark with the original delta-v penalties.
pub fn sagas(x: &[f64]) -> f64 {
    if x.len() < 12 {
        return PENALTY;
    }
    let mut problem = DsmProblem::new(MissionType::TimeToAus, &[3, 3, 5]);
    problem.au_distance = 50.0;
    problem.total_delta_v_limit = 6.782;
    problem.onboard_delta_v_limit = 1.782;
    let (mut objective, delta_v) = mga_dsm(x, &problem);
    let total: f64 = delta_v.iter().take(5).sum();
    let onboard = total - delta_v.first().copied().unwrap_or(0.0);
    if total > problem.total_delta_v_limit {
        objective += 10.0 + 10.0 * total;
    }
    if onboard > problem.onboard_delta_v_limit {
        objective += 10.0 + 10.0 * onboard;
    }
    sanitize(objective)
}

const ATLAS_VINF: [f64; 9] = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.75, 6.0];
const ATLAS_DECLINATION: [f64; 13] = [
    -40.0, -30.0, -29.0, -28.5, -20.0, -10.0, 0.0, 10.0, 20.0, 28.5, 29.0, 30.0, 40.0,
];
const ATLAS_MASS: [[f64; 9]; 13] = [
    [0.0; 9],
    [0.0; 9],
    [
        1160.0, 1100.0, 1010.0, 930.0, 830.0, 740.0, 630.0, 590.0, 550.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        2335.0, 2195.0, 2035.0, 1865.0, 1675.0, 1480.0, 1275.0, 1175.0, 1075.0,
    ],
    [
        1160.0, 1100.0, 1010.0, 930.0, 830.0, 740.0, 630.0, 590.0, 550.0,
    ],
    [0.0; 9],
    [0.0; 9],
];

fn interval(values: &[f64], value: f64) -> usize {
    values
        .windows(2)
        .position(|pair| pair[1] > value)
        .unwrap_or(values.len() - 2)
}

fn atlas_501(vinf: f64, declination: f64) -> f64 {
    if !(2.5..=6.0).contains(&vinf) || declination.abs() > 40.0 {
        return 0.0;
    }
    let velocity_index = interval(&ATLAS_VINF, vinf);
    let declination_index = interval(&ATLAS_DECLINATION, declination);
    let x0 = ATLAS_VINF[velocity_index];
    let x1 = ATLAS_VINF[velocity_index + 1];
    let y0 = ATLAS_DECLINATION[declination_index];
    let y1 = ATLAS_DECLINATION[declination_index + 1];
    let denominator = (x1 - x0) * (y1 - y0);
    (ATLAS_MASS[declination_index][velocity_index] * (x1 - vinf) * (y1 - declination)
        + ATLAS_MASS[declination_index][velocity_index + 1] * (vinf - x0) * (y1 - declination)
        + ATLAS_MASS[declination_index + 1][velocity_index] * (x1 - vinf) * (declination - y0)
        + ATLAS_MASS[declination_index + 1][velocity_index + 1] * (vinf - x0) * (declination - y0))
        / denominator
}

fn ecliptic_to_equatorial(vector: &Vec3) -> Vec3 {
    const INCLINATION: f64 = 0.409_072_975;
    [
        vector[0],
        vector[1] * INCLINATION.cos() - vector[2] * INCLINATION.sin(),
        vector[1] * INCLINATION.sin() + vector[2] * INCLINATION.cos(),
    ]
}

/// Unconstrained TandEM launcher-mass objective.
pub fn tandem_unconstrained(x: &[f64], sequence: &[usize]) -> f64 {
    if x.len() < 18 || sequence.len() != 5 || sequence.iter().any(|&body| !(1..=6).contains(&body))
    {
        return PENALTY;
    }
    let mut problem = DsmProblem::new(MissionType::OrbitInsertion, sequence);
    problem.periapsis = 80_330.0;
    problem.eccentricity = 0.985_314_079_963_58;
    let (_, delta_v) = mga_dsm(x, &problem);

    let (earth_position, earth_velocity) = planet_ephemerides(x[0], 3);
    let orbit_normal = unit(&cross(&earth_position, &earth_velocity));
    let tangent = unit(&earth_velocity);
    let transverse = cross(&orbit_normal, &tangent);
    let theta = 2.0 * PI * x[2];
    let phi = (2.0 * x[3] - 1.0).acos() - PI / 2.0;
    let ecliptic_vinf = [
        x[1] * (theta.cos() * phi.cos() * tangent[0]
            + theta.sin() * phi.cos() * transverse[0]
            + phi.sin() * orbit_normal[0]),
        x[1] * (theta.cos() * phi.cos() * tangent[1]
            + theta.sin() * phi.cos() * transverse[1]
            + phi.sin() * orbit_normal[1]),
        x[1] * (theta.cos() * phi.cos() * tangent[2]
            + theta.sin() * phi.cos() * transverse[2]
            + phi.sin() * orbit_normal[2]),
    ];
    let equatorial_vinf = ecliptic_to_equatorial(&ecliptic_vinf);
    let declination = (equatorial_vinf[2] / norm(&equatorial_vinf))
        .asin()
        .to_degrees();
    let initial_mass = atlas_501(x[1], declination);
    let onboard_delta_v: f64 = delta_v.iter().skip(1).take(5).sum::<f64>() + 0.165;
    sanitize(-initial_mass * (-onboard_delta_v / 312.0 / 9.806_65 * 1_000.0).exp())
}

/// TandEM objective with the original ten-year flight-time penalty.
pub fn tandem(x: &[f64], sequence: &[usize]) -> f64 {
    let mut value = tandem_unconstrained(x, sequence);
    if x.len() >= 8 {
        let flight_time: f64 = x[4..8].iter().sum();
        if flight_time > 3_652.5 {
            value += 1_000.0 * (flight_time - 3_652.5);
        }
    }
    sanitize(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn midpoint(lower: &[f64], upper: &[f64]) -> Vec<f64> {
        lower
            .iter()
            .zip(upper)
            .map(|(&lo, &hi)| 0.5 * (lo + hi))
            .collect()
    }

    fn relative_eq(actual: f64, expected: f64, tolerance: f64) {
        let error = (actual - expected).abs() / expected.abs().max(1.0);
        assert!(
            error <= tolerance,
            "actual={actual:.17}, expected={expected:.17}, relative error={error:e}"
        );
    }

    #[test]
    fn midpoint_reference_values_match_cpp() {
        let c1 = midpoint(
            &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        );
        relative_eq(cassini1(&c1), 206.132_108_003_359_01, 2.0e-9);

        let c2 = midpoint(
            &[
                -1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01,
                0.01, 1.05, 1.05, 1.15, 1.7, -PI, -PI, -PI, -PI,
            ],
            &[
                0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9,
                6.0, 6.0, 6.5, 291.0, PI, PI, PI, PI,
            ],
        );
        relative_eq(cassini2(&c2), 209.525_911_254_357_6, 1.0e-10);
    }

    #[test]
    fn all_dsm_reference_values_match_cpp() {
        let messenger_x = midpoint(
            &[
                1000.0, 1.0, 0.0, 0.0, 200.0, 30.0, 30.0, 30.0, 0.01, 0.01, 0.01, 0.01, 1.1, 1.1,
                1.1, -PI, -PI, -PI,
            ],
            &[
                4000.0, 5.0, 1.0, 1.0, 400.0, 400.0, 400.0, 400.0, 0.99, 0.99, 0.99, 0.99, 6.0,
                6.0, 6.0, PI, PI, PI,
            ],
        );
        relative_eq(messenger(&messenger_x), 107.657_527_271_420_46, 1.0e-10);
        let rosetta_x = midpoint(
            &[
                1460.0, 3.0, 0.0, 0.0, 300.0, 150.0, 150.0, 300.0, 700.0, 0.01, 0.01, 0.01, 0.01,
                0.01, 1.05, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI,
            ],
            &[
                1825.0, 5.0, 1.0, 1.0, 500.0, 800.0, 800.0, 800.0, 1850.0, 0.9, 0.9, 0.9, 0.9, 0.9,
                9.0, 9.0, 9.0, 9.0, PI, PI, PI, PI,
            ],
        );
        relative_eq(rosetta(&rosetta_x), 119.332_237_646_317_85, 1.0e-10);

        let messenger_full_x = midpoint(
            &[
                1900.0, 3.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 1.1, 1.1, 1.05, 1.05, 1.05, -PI, -PI, -PI, -PI, -PI,
            ],
            &[
                2200.0, 4.05, 1.0, 1.0, 500.0, 500.0, 500.0, 500.0, 500.0, 550.0, 0.99, 0.99, 0.99,
                0.99, 0.99, 0.99, 6.0, 6.0, 6.0, 6.0, 6.0, PI, PI, PI, PI, PI,
            ],
        );
        relative_eq(
            messenger_full(&messenger_full_x),
            281.568_694_637_097,
            2.0e-8,
        );

        let sagas_x = midpoint(
            &[
                7000.0, 0.0, 0.0, 0.0, 50.0, 300.0, 0.01, 0.01, 1.05, 8.0, -PI, -PI,
            ],
            &[
                9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9, 0.9, 7.0, 500.0, PI, PI,
            ],
        );
        relative_eq(sagas(&sagas_x), 101_606.949_970_038_08, 1.0e-8);
    }

    #[test]
    fn mga_and_minlp_reference_values_match_cpp() {
        let cassini = midpoint(
            &[-1000.0, 30.0, 100.0, 30.0, 400.0, 1000.0],
            &[0.0, 400.0, 470.0, 400.0, 2000.0, 6000.0],
        );
        let mut minlp = cassini.clone();
        minlp.extend([2.0, 2.0, 3.0, 5.0]);
        let (value, launch) = cassini1_minlp(&minlp);
        relative_eq(value, 206.132_108_003_359_01, 2.0e-9);
        relative_eq(launch, 17.188_975_524_720_508, 1.0e-10);

        let gtoc_x = midpoint(
            &[3000.0, 14.0, 14.0, 14.0, 14.0, 100.0, 366.0, 300.0],
            &[
                10000.0, 2000.0, 2000.0, 2000.0, 2000.0, 9000.0, 9000.0, 9000.0,
            ],
        );
        relative_eq(gtoc1(&gtoc_x), 1_999_999.999_999_997_2, 1.0e-12);
    }

    #[test]
    fn tandem_and_launcher_match_cpp() {
        let x = midpoint(
            &[
                5475.0, 2.5, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0, 0.01, 0.01, 0.01, 0.01, 1.05, 1.05,
                1.05, -PI, -PI, -PI,
            ],
            &[
                9132.0, 4.9, 1.0, 1.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.99, 0.99, 0.99, 0.99,
                10.0, 10.0, 10.0, PI, PI, PI,
            ],
        );
        let unconstrained = tandem_unconstrained(&x, &[3, 2, 3, 3, 6]);
        assert!(unconstrained < 0.0 && unconstrained.abs() < 1.0e-18);
        relative_eq(tandem(&x, &[3, 2, 3, 3, 6]), 1_387_500.0, 1.0e-14);
        assert_eq!(atlas_501(2.0, 0.0), 0.0);
        assert_eq!(atlas_501(4.0, 50.0), 0.0);
        assert!(atlas_501(4.0, 0.0) > 1_000.0);
    }

    #[test]
    fn invalid_lengths_return_penalty() {
        assert_eq!(cassini1(&[]), PENALTY);
        assert_eq!(cassini2(&[]), PENALTY);
        assert_eq!(cassini2_minlp(&[]), PENALTY);
        assert_eq!(messenger(&[]), PENALTY);
        assert_eq!(messenger_full(&[]), PENALTY);
        assert_eq!(rosetta(&[]), PENALTY);
        assert_eq!(sagas(&[]), PENALTY);
        assert_eq!(gtoc1(&[]), PENALTY);
        assert_eq!(cassini1_minlp(&[]), (PENALTY, PENALTY));
    }

    #[test]
    fn orbital_helpers_cover_elliptic_and_hyperbolic_paths() {
        relative_eq(
            mean_to_eccentric(0.5, 0.1),
            0.552_479_986_906_570_4,
            1.0e-13,
        );
        let hyperbolic = mean_to_eccentric(0.2, 1.5);
        assert!(hyperbolic.is_finite());
        let (position, velocity) = planet_ephemerides(0.0, 3);
        assert!(norm(&position) > 1.0e8);
        assert!(norm(&velocity) > 20.0);
        assert!(planet_ephemerides(0.0, 99).0[0].is_nan());
    }

    #[test]
    fn helper_edge_paths_are_finite_or_penalized() {
        assert_eq!(sanitize(f64::NAN), PENALTY);
        assert_eq!(bisect_root(1.0, 2.0, |value| value - 1.0), 1.0);
        assert_eq!(bisect_root(1.0, 2.0, |value| value - 2.0), 2.0);
        assert_eq!(bisect_root(1.0, 2.0, |value| value + 1.0), 0.0);
        assert!(lambert(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 0.0, 1.0, false).0[0].is_nan());

        for planet in 7..=9 {
            let (position, velocity) = planet_ephemerides(4_000.0, planet);
            assert!(
                position
                    .iter()
                    .chain(&velocity)
                    .all(|value| value.is_finite())
            );
        }

        let elements = [-2.0, 1.5, 0.2, 0.3, 0.4, 0.1];
        let (position, velocity) = full_elements_to_state(&elements, 1.0);
        assert!(
            position
                .iter()
                .chain(&velocity)
                .all(|value| value.is_finite())
        );
        let propagated = propagate_kepler(&position, &velocity, 0.1, 1.0);
        assert!(
            propagated
                .0
                .iter()
                .chain(&propagated.1)
                .all(|value| value.is_finite())
        );
        assert_eq!(
            time_to_distance(&[2.0, 0.0, 0.0], &[0.0, 1.0, 0.0], 1.0),
            12.0
        );
    }

    #[test]
    fn cassini2_minlp_accepts_and_validates_planet_sequence() {
        let mut x = midpoint(
            &[
                -1000.0, 3.0, 0.0, 0.0, 100.0, 100.0, 30.0, 400.0, 800.0, 0.01, 0.01, 0.01, 0.01,
                0.01, 1.05, 1.05, 1.15, 1.7, -PI, -PI, -PI, -PI,
            ],
            &[
                0.0, 5.0, 1.0, 1.0, 400.0, 500.0, 300.0, 1600.0, 2200.0, 0.9, 0.9, 0.9, 0.9, 0.9,
                6.0, 6.0, 6.5, 291.0, PI, PI, PI, PI,
            ],
        );
        x.extend([2.0, 2.0, 3.0, 5.0]);
        assert!(cassini2_minlp(&x).is_finite());
        x[22] = 7.0;
        assert_eq!(cassini2_minlp(&x), PENALTY);
    }
}
