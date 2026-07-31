//! L5: a ZDT1 front is measured, not only plotted.

use fcmaes_core::{
    Fitness, Mode, ModeParams, ReferencePoint, hypervolume, igd_plus, pareto_indices,
};

use crate::suites::Suite;
use crate::suites::zdt::Zdt;

pub fn run(_workers: i32) -> Result<String, String> {
    let problem = Zdt::Zdt1(30);
    let (lower, upper) = problem.bounds();
    let fitness = Fitness::bounded(30, 2, &lower, &upper);
    let mut mode = Mode::try_new(
        fitness,
        2,
        0,
        None,
        &ModeParams {
            popsize: 64,
            seed: 42,
            ..Default::default()
        },
    )
    .map_err(str::to_owned)?;
    for _ in 0..100 {
        let decisions = mode.ask();
        let values = decisions
            .iter()
            .map(|x| problem.evaluate(x).expect("MODE honors bounds"))
            .collect::<Vec<_>>();
        mode.try_tell(&values).map_err(str::to_owned)?;
    }
    let values = mode
        .population()
        .iter()
        .map(|x| problem.evaluate(x).expect("MODE honors bounds"))
        .collect::<Vec<_>>();
    let front = pareto_indices(&values, 2)
        .map_err(str::to_owned)?
        .into_iter()
        .map(|index| values[index].clone())
        .collect::<Vec<_>>();
    let coordinates: Vec<f64> = (0..2)
        .map(|axis| {
            let minimum = front
                .iter()
                .map(|point| point[axis])
                .fold(f64::INFINITY, f64::min);
            let maximum = front
                .iter()
                .map(|point| point[axis])
                .fold(f64::NEG_INFINITY, f64::max);
            maximum + 0.1 * (maximum - minimum).max(1.0)
        })
        .collect();
    let reference_point = ReferencePoint::new(coordinates).map_err(|error| error.to_string())?;
    let hv = hypervolume(&front, &reference_point).map_err(|error| error.to_string())?;
    let fixed_reference = problem.reference_point().unwrap();
    let fixed_outside = front
        .iter()
        .filter(|point| {
            point
                .iter()
                .zip(fixed_reference.as_slice())
                .any(|(value, reference)| value > reference)
        })
        .count();
    let fixed_status = if fixed_outside == 0 {
        "exact"
    } else {
        "not-applicable"
    };
    let igd = igd_plus(&front, &problem.reference_front(501).unwrap())
        .map_err(|error| error.to_string())?;
    Ok(format!(
        "L5 measured front | points={} hv={:.6e} fixed_status={} fixed_outside={} igd_plus={igd:.6e}\n",
        front.len(),
        hv.estimate.value(),
        fixed_status,
        fixed_outside
    ))
}
