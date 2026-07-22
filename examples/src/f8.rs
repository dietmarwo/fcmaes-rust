//! F-8 aircraft bang-bang control objective from `examples/f8.py`.

use crate::integration::dopri5;

pub const KSI: f64 = 0.05236;

pub fn derivative(state: &[f64; 3], control: f64) -> [f64; 3] {
    let [y0, y1, y2] = *state;
    [
        -0.877 * y0 + y2 - 0.088 * y0 * y2 + 0.47 * y0.powi(2)
            - 0.019 * y1.powi(2)
            - y0.powi(2) * y2
            + 3.846 * y0.powi(3)
            + 0.215 * KSI
            - 0.28 * y0.powi(2) * KSI
            + 0.47 * y0 * KSI.powi(2)
            - 0.63 * KSI.powi(3)
            - (0.215 * KSI - 0.28 * y0.powi(2) * KSI - 0.63 * KSI.powi(3)) * 2.0 * control,
        y2,
        -4.208 * y0 - 0.396 * y2 - 0.47 * y0.powi(2) - 3.564 * y0.powi(3) + 20.967 * KSI
            - 6.265 * y0.powi(2) * KSI
            + 46.0 * y0 * KSI.powi(2)
            - 61.4 * KSI.powi(3)
            - (20.967 * KSI - 6.265 * y0.powi(2) * KSI - 61.4 * KSI.powi(3)) * 2.0 * control,
    ]
}

pub fn final_state(switch_durations: &[f64]) -> Result<[f64; 3], &'static str> {
    let mut state = [0.4655, 0.0, 0.0];
    for (index, &duration) in switch_durations.iter().enumerate() {
        if !(0.0..=2.0).contains(&duration) {
            return Err("F-8 switch durations must be in [0, 2]");
        }
        if duration == 0.0 {
            continue;
        }
        // Python: w = (i + 1) % 2, hence 1, 0, 1, ...
        let control = ((index + 1) % 2) as f64;
        state = dopri5(state, 0.0, duration, 0.05, 1e-10, 1e-10, |_, y| {
            derivative(y, control)
        })?;
    }
    Ok(state)
}

pub fn fitness(x: &[f64]) -> f64 {
    let Ok(state) = final_state(x) else {
        return 1e10;
    };
    0.1 * x.iter().sum::<f64>() + state.iter().map(|value| value.abs()).sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_reference() {
        let result = derivative(&[0.4655, 0.0, 0.0], 1.0);
        let expected = [0.07415399137846734, 0.0, -3.3793975774875316];
        for (actual, expected) in result.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14, "{actual} != {expected}");
        }
    }

    #[test]
    fn objective_matches_scipy_reference() {
        let x = [0.3, 0.7, 0.2, 0.6, 0.4, 0.1];
        let state = final_state(&x).unwrap();
        let expected = [
            0.076_618_927_323_909_3,
            -0.46488993509127313,
            0.026233880732298837,
        ];
        for (actual, expected) in state.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-8, "{actual} != {expected}");
        }
        assert!((fitness(&x) - 0.7977427431474813).abs() < 3e-8);
    }

    #[test]
    fn invalid_duration_is_penalized() {
        assert_eq!(fitness(&[2.1]), 1e10);
    }
}
