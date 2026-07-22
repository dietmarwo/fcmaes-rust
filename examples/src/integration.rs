//! Allocation-free Dormand-Prince 5(4) integration for the ODE examples.

pub(crate) fn dopri5<const N: usize>(
    mut state: [f64; N],
    start: f64,
    end: f64,
    initial_step: f64,
    atol: f64,
    rtol: f64,
    mut derivative: impl FnMut(f64, &[f64; N]) -> [f64; N],
) -> Result<[f64; N], &'static str> {
    if end < start || !start.is_finite() || !end.is_finite() {
        return Err("integration targets must be finite and nondecreasing");
    }
    if end == start {
        return Ok(state);
    }
    let mut time = start;
    let mut step = initial_step.min(end - start).max(1e-12);
    for _ in 0..2_000_000 {
        step = step.min(end - time);
        let k1 = derivative(time, &state);
        let k2 = derivative(
            time + step / 5.0,
            &combine(&state, step, &[(&k1, 1.0 / 5.0)]),
        );
        let k3 = derivative(
            time + 3.0 * step / 10.0,
            &combine(&state, step, &[(&k1, 3.0 / 40.0), (&k2, 9.0 / 40.0)]),
        );
        let k4 = derivative(
            time + 4.0 * step / 5.0,
            &combine(
                &state,
                step,
                &[(&k1, 44.0 / 45.0), (&k2, -56.0 / 15.0), (&k3, 32.0 / 9.0)],
            ),
        );
        let k5 = derivative(
            time + 8.0 * step / 9.0,
            &combine(
                &state,
                step,
                &[
                    (&k1, 19372.0 / 6561.0),
                    (&k2, -25360.0 / 2187.0),
                    (&k3, 64448.0 / 6561.0),
                    (&k4, -212.0 / 729.0),
                ],
            ),
        );
        let k6 = derivative(
            time + step,
            &combine(
                &state,
                step,
                &[
                    (&k1, 9017.0 / 3168.0),
                    (&k2, -355.0 / 33.0),
                    (&k3, 46732.0 / 5247.0),
                    (&k4, 49.0 / 176.0),
                    (&k5, -5103.0 / 18656.0),
                ],
            ),
        );
        let fifth = combine(
            &state,
            step,
            &[
                (&k1, 35.0 / 384.0),
                (&k3, 500.0 / 1113.0),
                (&k4, 125.0 / 192.0),
                (&k5, -2187.0 / 6784.0),
                (&k6, 11.0 / 84.0),
            ],
        );
        let k7 = derivative(time + step, &fifth);
        let fourth = combine(
            &state,
            step,
            &[
                (&k1, 5179.0 / 57600.0),
                (&k3, 7571.0 / 16695.0),
                (&k4, 393.0 / 640.0),
                (&k5, -92097.0 / 339200.0),
                (&k6, 187.0 / 2100.0),
                (&k7, 1.0 / 40.0),
            ],
        );
        let error = (0..N).fold(0.0_f64, |maximum, i| {
            let scale = atol + rtol * state[i].abs().max(fifth[i].abs());
            maximum.max((fifth[i] - fourth[i]).abs() / scale)
        });
        if error <= 1.0 {
            state = fifth;
            time += step;
            if time >= end {
                return state
                    .iter()
                    .all(|value| value.is_finite())
                    .then_some(state)
                    .ok_or("non-finite ODE state");
            }
        }
        let factor = if error == 0.0 {
            5.0
        } else {
            (0.9 * error.powf(-0.2)).clamp(0.1, 5.0)
        };
        step *= factor;
        if step < 1e-14 {
            return Err("ODE step underflow");
        }
    }
    Err("ODE step limit exceeded")
}

fn combine<const N: usize>(base: &[f64; N], step: f64, terms: &[(&[f64; N], f64)]) -> [f64; N] {
    std::array::from_fn(|i| {
        base[i]
            + step
                * terms
                    .iter()
                    .map(|(values, factor)| factor * values[i])
                    .sum::<f64>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_exponential() {
        let state = dopri5([1.0], 0.0, 1.0, 0.1, 1e-12, 1e-12, |_, y| [y[0]]).unwrap();
        assert!((state[0] - std::f64::consts::E).abs() < 1e-10);
    }
}
