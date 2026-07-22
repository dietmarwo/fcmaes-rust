use fcmaes_examples::problems;
use fcmaes_examples::runner::{Cli, run_basic};

fn main() {
    let cli = Cli::from_env();
    let selected = problems::selected(cli.problem.as_deref()).unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(2);
    });
    for problem in selected {
        let result = run_basic(&problem, &cli);
        println!(
            "{}: value={:.12} evaluations={} runs={} x={:?}",
            problem.name, result.y, result.evaluations, result.runs, result.x
        );
    }
}
