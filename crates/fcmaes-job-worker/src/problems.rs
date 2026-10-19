//! Registered native benchmark models of the multi-objective and
//! quality-diversity profiles. Scalar retry problems keep using the shared
//! legacy registry, so `fcmaes.retry/1` and `fcmaes.job/0.1` stay identical.
use optimization_job_api::{Result, legacy};
use serde_json::{Value, json};
use std::f64::consts::PI;

/// A registered multi-objective model: `nobj` minimized objectives followed
/// by `ncon` constraints, feasible when every constraint is at most zero.
pub struct MultiObjective {
    pub id: &'static str,
    pub nobj: usize,
    pub ncon: usize,
    /// The decision box job bounds must lie inside.
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// Reference point used for the hypervolume when the job gives none.
    pub reference_point: Vec<f64>,
    pub evaluate: fn(&[f64]) -> Vec<f64>,
}

/// A registered quality-diversity model: minimized fitness plus a
/// two-dimensional behavior descriptor with fixed bounds.
pub struct QualityDiversity {
    pub id: &'static str,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub descriptor_lower: Vec<f64>,
    pub descriptor_upper: Vec<f64>,
    pub evaluate: fn(&[f64]) -> (f64, Vec<f64>),
}

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

fn check_version(problem: &legacy::Problem) -> Result<usize> {
    if problem.version != "1" {
        return Err("unsupported problem version".into());
    }
    let n = problem.config.dimension;
    if !(1..=10000).contains(&n) {
        return Err("dimension must be 1..10000".into());
    }
    Ok(n)
}

/// Validates a multi-objective problem and expands its bounds.
pub fn multi_objective(mut problem: legacy::Problem) -> Result<(legacy::Problem, MultiObjective)> {
    let n = check_version(&problem)?;
    let model = match problem.id.as_str() {
        "benchmark.zdt1" if n >= 2 => MultiObjective {
            id: "benchmark.zdt1",
            nobj: 2,
            ncon: 0,
            lower: vec![0.0; n],
            upper: vec![1.0; n],
            reference_point: vec![1.1, 1.1],
            evaluate: zdt1,
        },
        "benchmark.dtlz2" if n >= 3 => MultiObjective {
            id: "benchmark.dtlz2",
            nobj: 3,
            ncon: 0,
            lower: vec![0.0; n],
            upper: vec![1.0; n],
            reference_point: vec![1.1, 1.1, 1.1],
            evaluate: dtlz2,
        },
        "benchmark.binh-korn" if n == 2 => MultiObjective {
            id: "benchmark.binh-korn",
            nobj: 2,
            ncon: 2,
            lower: vec![0.0, 0.0],
            upper: vec![5.0, 3.0],
            reference_point: vec![140.0, 50.0],
            evaluate: binh_korn,
        },
        _ => return Err("unknown multi-objective problem or invalid dimension".into()),
    };
    problem.bounds = problem.bounds.resolve(n)?;
    if !problem.bounds.within(&model.lower, &model.upper) {
        return Err(format!("{} bounds must lie inside its canonical box", model.id).into());
    }
    Ok((problem, model))
}

/// Validates a quality-diversity problem and expands its bounds.
pub fn quality_diversity(
    mut problem: legacy::Problem,
) -> Result<(legacy::Problem, QualityDiversity)> {
    let n = check_version(&problem)?;
    let evaluate: fn(&[f64]) -> (f64, Vec<f64>) = match problem.id.as_str() {
        "benchmark.sphere-projection" => sphere_projection,
        "benchmark.rastrigin-projection" => rastrigin_projection,
        _ => return Err("unknown quality-diversity problem".into()),
    };
    if n < 2 || !n.is_multiple_of(2) {
        return Err("projection problems need an even dimension of at least 2".into());
    }
    let reach = SPAN * (n / 2) as f64;
    let model = QualityDiversity {
        id: if problem.id == "benchmark.sphere-projection" {
            "benchmark.sphere-projection"
        } else {
            "benchmark.rastrigin-projection"
        },
        lower: vec![-SPAN; n],
        upper: vec![SPAN; n],
        descriptor_lower: vec![-reach; 2],
        descriptor_upper: vec![reach; 2],
        evaluate,
    };
    problem.bounds = problem.bounds.resolve(n)?;
    if !problem.bounds.within(&model.lower, &model.upper) {
        return Err(format!("{} bounds must lie inside [-5.12, 5.12]", model.id).into());
    }
    Ok((problem, model))
}

/// Discovery entries of the registered multi-objective and QD models.
pub fn discovery() -> (Value, Value) {
    (
        json!([
            {"id":"benchmark.zdt1","version":"1","dimensions":"2..10000","objectives":2,"constraints":0,
             "box":[0.0,1.0],"reference_point":[1.1,1.1],"front":"f2 = 1 - sqrt(f1), f1 in [0,1]"},
            {"id":"benchmark.dtlz2","version":"1","dimensions":"3..10000","objectives":3,"constraints":0,
             "box":[0.0,1.0],"reference_point":[1.1,1.1,1.1],"front":"f1^2 + f2^2 + f3^2 = 1, f >= 0"},
            {"id":"benchmark.binh-korn","version":"1","dimensions":2,"objectives":2,"constraints":2,
             "lower":[0.0,0.0],"upper":[5.0,3.0],"reference_point":[140.0,50.0],
             "feasible":"every constraint value <= 0"}
        ]),
        json!([
            {"id":"benchmark.sphere-projection","version":"1","dimensions":"even, 2..10000",
             "box":[-5.12,5.12],"fitness":"sum((x_i - 2.048)^2), minimized",
             "descriptor":"sums of the clipped first and second halves of x","descriptor_bounds":"[-5.12*n/2, 5.12*n/2] each"},
            {"id":"benchmark.rastrigin-projection","version":"1","dimensions":"even, 2..10000",
             "box":[-5.12,5.12],"fitness":"10n + sum(z^2 - 10 cos(2 pi z)), z = x - 2.048, minimized",
             "descriptor":"sums of the clipped first and second halves of x","descriptor_bounds":"[-5.12*n/2, 5.12*n/2] each"}
        ]),
    )
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
