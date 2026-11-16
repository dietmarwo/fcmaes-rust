//! Fixed-seed compatibility records for the cooperative-stop release.
use fcmaes_core::*;
fn main() {
    let objective = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    for seed in 1..=20 {
        let fit = || Fitness::bounded(4, 1, &[-5.; 4], &[5.; 4]);
        macro_rules! record {
            ($name:expr,$r:expr) => {{
                let r = $r;
                println!(
                    "{} {} {:?} {:016x} {} {} {}",
                    $name,
                    seed,
                    r.x,
                    r.y.to_bits(),
                    r.evaluations,
                    r.iterations,
                    r.stop
                );
            }};
        }
        let p = BiteParams {
            seed,
            max_evaluations: 1000,
            ..Default::default()
        };
        record!(
            "bite",
            BiteOpt::new(&[-5.; 4], &[5.; 4], None, &p).optimize(&objective)
        );
        record!(
            "deep",
            DeepBiteOpt::new(&[-5.; 4], &[5.; 4], None, &p, 3).optimize(&objective)
        );
        let p = DeParams {
            seed,
            max_evaluations: 1000,
            ..Default::default()
        };
        record!(
            "de",
            De::new(fit(), &[], &[], None, &p).optimize(&objective)
        );
        record!(
            "delayed",
            De::new(fit(), &[], &[], None, &p).optimize_delayed(&objective, 4)
        );
        record!(
            "cmaes",
            Cmaes::new(
                fit(),
                &[0.; 4],
                &[0.3],
                &CmaesParams {
                    seed,
                    popsize: 8,
                    max_evaluations: 1000,
                    ..Default::default()
                }
            )
            .optimize(&objective, 1)
        );
        record!(
            "crfmnes",
            Crfmnes::new(
                fit(),
                &[0.; 4],
                0.3,
                &CrfmnesParams {
                    seed,
                    popsize: 8,
                    max_evaluations: 1000,
                    ..Default::default()
                }
            )
            .optimize_batch(|xs| xs.iter().map(|x| objective(x)).collect())
        );
        record!(
            "pgpe",
            Pgpe::new(
                fit(),
                &[0.; 4],
                &[0.3],
                &PgpeParams {
                    seed,
                    popsize: 8,
                    max_evaluations: 1000,
                    ..Default::default()
                }
            )
            .optimize_batch(|xs| xs.iter().map(|x| objective(x)).collect())
        );
    }
}
