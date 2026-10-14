//! Constrained MODE with a separate exact archive of every feasible evaluation.
use fcmaes_core::{Fitness, Mode, ModeParams, ParetoArchive2D, ParetoBounds2D, ParetoScore};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mode = Mode::try_new(
        Fitness::bounded(2, 3, &[0.0; 2], &[1.0; 2]),
        2,
        1,
        None,
        &ModeParams {
            popsize: 32,
            seed: 42,
            ..Default::default()
        },
    )?;
    let mut archive = ParetoArchive2D::new(
        ParetoBounds2D::new([0.0; 2], [2.0; 2])?,
        ParetoScore::Linear,
    );
    let mut evaluations = 0;
    for _ in 0..40 {
        let xs = mode.try_ask()?;
        let ys: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                vec![
                    x.iter().map(|v| v * v).sum(),
                    x.iter().map(|v| (1.0 - v).powi(2)).sum(),
                    x.iter().sum::<f64>() - 1.5,
                ]
            })
            .collect();
        for (x, y) in xs.iter().zip(&ys) {
            if y.iter().all(|v| v.is_finite()) && y[2] <= 0.0 {
                archive.insert_with([y[0], y[1]], || x.clone())?;
            }
        }
        evaluations += ys.len();
        mode.try_tell(&ys)?;
    }
    println!(
        "evaluations={evaluations} population={} archive={} HV={:.12}",
        mode.popsize(),
        archive.len(),
        archive.hypervolume()
    );
    Ok(())
}
