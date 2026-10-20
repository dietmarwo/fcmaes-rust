//! The built-in benchmark models. The scalar ones evaluate the objectives of
//! the legacy `fcmaes.job/0.1` registry, so `fcmaes.retry/1` and
//! `fcmaes.job/0.1` runs from the same seed stay identical.
use crate::registry::{MAX_DIMENSION, MultiObjective, QualityDiversity, Registry, Scalar};
use fcmaes_native_worker::objective_by_id;
use optimization_job_api::legacy::{CASSINI1_LOWER, CASSINI1_UPPER};
use serde_json::json;
use std::f64::consts::PI;

/// ZDT1 (Zitzler, Deb and Thiele 2000): convex front `f2 = 1 - sqrt(f1)`.
pub fn zdt1(x: &[f64]) -> Vec<f64> {
    let f1 = x[0];
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    vec![f1, g * (1.0 - (f1 / g).sqrt())]
}

/// DTLZ2 with three objectives (Deb et al. 2002): the front is the positive
/// octant of the unit sphere.
pub fn dtlz2(x: &[f64]) -> Vec<f64> {
    let g: f64 = x[2..].iter().map(|v| (v - 0.5) * (v - 0.5)).sum();
    let (a, b) = (x[0] * PI / 2.0, x[1] * PI / 2.0);
    vec![
        (1.0 + g) * a.cos() * b.cos(),
        (1.0 + g) * a.cos() * b.sin(),
        (1.0 + g) * a.sin(),
    ]
}

/// Binh and Korn (1997): two objectives and two constraints, written as
/// `c <= 0` for feasibility.
pub fn binh_korn(x: &[f64]) -> Vec<f64> {
    let (a, b) = (x[0], x[1]);
    vec![
        4.0 * a * a + 4.0 * b * b,
        (a - 5.0).powi(2) + (b - 5.0).powi(2),
        (a - 5.0).powi(2) + b * b - 25.0,
        7.7 - (a - 8.0).powi(2) - (b + 3.0).powi(2),
    ]
}

/// Bound of the linear-projection benchmarks (Fontaine et al. 2020, CMA-ME).
const SPAN: f64 = 5.12;
/// Shift of their optimum, `0.4 * 5.12`.
const SHIFT: f64 = 0.4 * SPAN;

fn clip(v: f64) -> f64 {
    if v.abs() <= SPAN { v } else { SPAN / v }
}

/// Behavior descriptor of the linear-projection benchmarks: the sums of the
/// clipped first and second halves of `x`.
fn projection(x: &[f64]) -> Vec<f64> {
    let half = x.len() / 2;
    vec![
        x[..half].iter().copied().map(clip).sum(),
        x[half..].iter().copied().map(clip).sum(),
    ]
}

/// Shifted sphere with the linear-projection descriptor.
pub fn sphere_projection(x: &[f64]) -> (f64, Vec<f64>) {
    let f = x.iter().map(|v| (v - SHIFT) * (v - SHIFT)).sum();
    (f, projection(x))
}

/// Shifted Rastrigin with the linear-projection descriptor.
pub fn rastrigin_projection(x: &[f64]) -> (f64, Vec<f64>) {
    let f = 10.0 * x.len() as f64
        + x.iter()
            .map(|v| {
                let z = v - SHIFT;
                z * z - 10.0 * (2.0 * PI * z).cos()
            })
            .sum::<f64>();
    (f, projection(x))
}

/// Registers the built-in benchmark models: the scalar models of
/// `fcmaes.job/0.1` (with the same objectives and limits), three
/// multi-objective and two quality-diversity models.
pub fn register(registry: &mut Registry) {
    registry
        .scalar("benchmark.branin", "1", json!({"dimensions":2,"lower":[-5.0,0.0],"upper":[10.0,15.0],"minimum":0.39788735772973816}), |m| {
            m.require_dimension(2..=2)?;
            Ok(Scalar::new(fcmaes_surrogate::problems::branin).domain(vec![-5.0,0.0],vec![10.0,15.0]))
        })
        .scalar("benchmark.hartmann3", "1", json!({"dimensions":3,"minimum":-3.8627797873326624}), |m| {
            m.require_dimension(3..=3)?;
            Ok(Scalar::new(fcmaes_surrogate::problems::hartmann3).domain(vec![0.0;3],vec![1.0;3]))
        })
        .scalar(
            "benchmark.sphere",
            "1",
            json!({"dimensions":"1..10000","objective":"sum(x_i^2)","minimum":"0 at the origin"}),
            |_| Ok(Scalar::new(objective_by_id("benchmark.sphere"))),
        )
        .scalar(
            "benchmark.rosenbrock",
            "1",
            json!({"dimensions":"2..10000","minimum":"0 at (1, ..., 1)"}),
            |m| {
                m.require_dimension(2..=MAX_DIMENSION)?;
                Ok(Scalar::new(objective_by_id("benchmark.rosenbrock")))
            },
        )
        .scalar(
            "gtop.cassini1",
            "1",
            json!({"dimensions":6,"lower":CASSINI1_LOWER,"upper":CASSINI1_UPPER,
                "bounds":"must lie inside lower/upper"}),
            |m| {
                m.require_dimension(6..=6)?;
                Ok(Scalar::new(objective_by_id("gtop.cassini1"))
                    .domain(CASSINI1_LOWER.to_vec(), CASSINI1_UPPER.to_vec()))
            },
        )
        .multi_objective(
            "benchmark.zdt1",
            "1",
            json!({"dimensions":"2..10000","objectives":2,"constraints":0,
             "box":[0.0,1.0],"reference_point":[1.1,1.1],"front":"f2 = 1 - sqrt(f1), f1 in [0,1]"}),
            |m| {
                let n = m.require_dimension(2..=MAX_DIMENSION)?;
                Ok(MultiObjective::new(2, 0, zdt1)
                    .reference_point(vec![1.1, 1.1])
                    .domain(vec![0.0; n], vec![1.0; n]))
            },
        )
        .multi_objective(
            "benchmark.dtlz2",
            "1",
            json!({"dimensions":"3..10000","objectives":3,"constraints":0,
             "box":[0.0,1.0],"reference_point":[1.1,1.1,1.1],"front":"f1^2 + f2^2 + f3^2 = 1, f >= 0"}),
            |m| {
                let n = m.require_dimension(3..=MAX_DIMENSION)?;
                Ok(MultiObjective::new(3, 0, dtlz2)
                    .reference_point(vec![1.1, 1.1, 1.1])
                    .domain(vec![0.0; n], vec![1.0; n]))
            },
        )
        .multi_objective(
            "benchmark.binh-korn",
            "1",
            json!({"dimensions":2,"objectives":2,"constraints":2,
             "lower":[0.0,0.0],"upper":[5.0,3.0],"reference_point":[140.0,50.0],
             "feasible":"every constraint value <= 0"}),
            |m| {
                m.require_dimension(2..=2)?;
                Ok(MultiObjective::new(2, 2, binh_korn)
                    .reference_point(vec![140.0, 50.0])
                    .domain(vec![0.0, 0.0], vec![5.0, 3.0]))
            },
        );
    for (id, evaluate, fitness) in [
        (
            "benchmark.sphere-projection",
            sphere_projection as fn(&[f64]) -> (f64, Vec<f64>),
            "sum((x_i - 2.048)^2), minimized",
        ),
        (
            "benchmark.rastrigin-projection",
            rastrigin_projection,
            "10n + sum(z^2 - 10 cos(2 pi z)), z = x - 2.048, minimized",
        ),
    ] {
        registry.quality_diversity(
            id,
            "1",
            json!({"dimensions":"even, 2..10000","box":[-SPAN,SPAN],"fitness":fitness,
                "descriptor":"sums of the clipped first and second halves of x",
                "descriptor_bounds":"[-5.12*n/2, 5.12*n/2] each"}),
            move |m| {
                let n = m.dimension();
                if n < 2 || !n.is_multiple_of(2) {
                    return Err("projection problems need an even dimension of at least 2".into());
                }
                let reach = SPAN * (n / 2) as f64;
                Ok(
                    QualityDiversity::new(vec![-reach; 2], vec![reach; 2], evaluate)
                        .domain(vec![-SPAN; n], vec![SPAN; n]),
                )
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_points_have_known_values() {
        assert_eq!(zdt1(&[0.25, 0.0, 0.0]), vec![0.25, 0.5]);
        let d = dtlz2(&[0.0, 0.0, 0.5, 0.5]);
        assert!((d[0] - 1.0).abs() < 1e-15 && d[1].abs() < 1e-15 && d[2].abs() < 1e-15);
        assert_eq!(
            binh_korn(&[0.0, 0.0]),
            vec![0.0, 50.0, 0.0, 7.7 - 64.0 - 9.0]
        );
        let (f, d) = sphere_projection(&[SHIFT; 4]);
        assert_eq!(f, 0.0);
        assert_eq!(d, vec![2.0 * SHIFT, 2.0 * SHIFT]);
        assert_eq!(rastrigin_projection(&[SHIFT; 2]).0, 0.0);
        assert_eq!(clip(10.24), 0.5);
    }
}
