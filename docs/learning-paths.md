# Learning paths

Three tracks order the existing tutorials of the
[20 related repositories](repositories.md) by reader. Each has five stages
and ends in a capstone that continuous integration runs. The overview sorts
the same repositories [by problem type](repositories.md#where-to-start). An
agent that has to choose a solver for a problem in hand starts at the shared
[solver-selection guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)
instead.

Every stage links its material and names a checkpoint: a command whose
output shows that the stage worked. Install the prerequisites from the linked
setup page first, and run the checkpoint from the root of the repository it
names. `out/` in a command is a new directory that you choose.

## Python user

Start here with a Python objective or data to fit. No Rust programming is
required; the capstone builds existing Rust executables on Linux.

| Stage | Goal and material | Checkpoint |
| --- | --- | --- |
| P1 | Minimize a bounded objective with [parallel retry](https://github.com/dietmarwo/fast-cma-es/blob/master/ai-context.md) and the [Python tutorials](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/README.adoc). | After installing `fcmaes`, `python -c "from fcmaes import retry; from scipy.optimize import Bounds; r=retry.minimize(lambda x: sum(v*v for v in x), Bounds([-5]*3,[5]*3), num_retries=8, max_evaluations=2000); print(r.fun,r.nfev)"` ends with the best value and the number of evaluations used. The retries run on all available CPUs. |
| P2 | Score a population natively: [hydro tutorial](https://github.com/dietmarwo/hydro-craft/blob/main/docs/tutorial.md) and [examples](https://github.com/dietmarwo/hydro-craft/blob/main/examples/python/README.md). | In hydro-craft, `python examples/python/fcmaes_cmaes_population.py` prints a fit and held-out metrics. Find the graded penalty that it assigns to invalid rows. |
| P3 | Use linear, integer or smooth constrained structure: [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). | In optimization-jobs, run `python tutorials/structured/run.py --chapter LP1 --output out/lp1`, then the same command with `--chapter SQP1 --output out/sqp1`. Both summaries contain independent checks. The [solver-choice exercise](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/solver-choice/README.md) explains when such structure makes LP/IP or SQP the right tool. |
| P4 | Calibrate a stochastic simulator: [CRN guide](https://github.com/dietmarwo/CRN-craft/blob/main/docs/guide/01-install.md), [C1 lesson](https://github.com/dietmarwo/CRN-craft/blob/main/tutorials/optimization/C1-calibration.md), [simulator setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/simulators/README.md). | In CRN-craft, `python tutorials/optimization/run.py C1 --output out/c1` fits on fixed training paths and reports held-out losses from a disjoint root seed. |
| P5 | **Capstone:** a repeatable [hydro command job](https://github.com/dietmarwo/hydro-craft/blob/main/docs/command-jobs.md), followed by the [domain campaign](https://github.com/dietmarwo/experiment-craft/blob/main/examples/campaigns/domain-evaluators/README.md). | Work through sections 1 to 5 of the walkthrough in its workspace. The five-seed campaign of section 5 writes `hydro-five-seeds/checks.json` with the training, validation and test error of every arm, and `hydro-five-seeds/report/report.md` with the state of every job. Select an arm by its validation error. |

From P5, move to [agent stage A2](#agent-builder) to expose registered workers
over MCP. Command jobs themselves still need a shell.

## Rust user

Start here when you write the objective or the simulator in Rust. Run Cargo
with the linked project's toolchain and locked dependencies.

| Stage | Goal and material | Checkpoint |
| --- | --- | --- |
| R1 | First objective: [getting started](getting-started.md), [one-shot and ask/tell](optimizers.md). | Run the bounded-sphere program from getting started in your own crate with `cargo run --release`; it prints the value, the evaluations and the coordinates. Repeat it with the ask/evaluate/tell loop from the optimizer guide on the same bounds. |
| R2 | Choose a method and a budget: [chooser](choosing-an-optimizer.md), [foundations](../foundations/README.md), [retry](retry.md). | `cargo run --release --locked --manifest-path foundations/Cargo.toml -- --lesson 3 --workers 2` reports `identical=true` for serial and parallel retry. State your method and evaluation budget for a problem of your own before you change either. |
| R3 | Run an application: [tutorial index](../tutorials/README.md), starting with the [Rapier trebuchet](../tutorials/rapier-trebuchet/README.md), then [GTOC1](../tutorials/gtoc1/README.md). | `cargo run --release --locked --manifest-path tutorials/rapier-trebuchet/Cargo.toml -- --mode simulate --no-output --x 3.5,80,4,2.5,-0.75,-0.05,2,2` prints one deterministic physical replay. Continue with the tutorial's smoke and recorded-result commands. |
| R4 | Solve structured models in Rust: [LP API](https://github.com/dietmarwo/LP-IP-craft/blob/main/docs/api.md), [LP example](https://github.com/dietmarwo/LP-IP-craft/blob/main/crates/native/examples/solve.rs), [SQP example](https://github.com/dietmarwo/SQP-craft/blob/main/crates/sqp/examples/solve.rs). | In LP-IP-craft, `cargo run --release --locked -p lp-ip-craft-native --example solve` solves a binary model and prints its independent evaluation. In SQP-craft, `cargo run --release --locked -p sqp-craft-solver --example solve` prints a converged solution and its KKT residuals. |
| R5 | **Capstone:** [register a model](job-api/models.md), then [run the native job tutorial](../tutorials/job-api/README.md); see also the [application workers](job-api/application-tutorials.md). | Build the runner, the worker and `custom_worker` as the tutorial shows. Run `export PATH="$PWD/target/release:$PATH"` in fcmaes-rust before the model guide's bare runner commands. Then `python3 tutorials/job-api/run.py --root out/native-jobs` runs the registered curve-fit model and writes `tutorial-report.json` with independent checks. |

The cross-solver chapters X1 to X4 are taught through the Python APIs; for
mixed formulations, see [Python stage P3](#python-user). For an agent that
uses the worker from R5, continue at [A2](#agent-builder).

## Agent builder

Start here when an AI agent is to choose a solver, run it and report the
result correctly. A worker's discovery output determines what the agent can
submit, and the evidence in a result determines what it may claim.

| Stage | Goal and material | Checkpoint |
| --- | --- | --- |
| A1 | Choose from a problem card: [agent entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md), [solver-choice lesson](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/solver-choice/README.md). | In optimization-jobs, after the structured setup, `python tutorials/structured/run.py --chapter X1 --output out/choice` checks an LP, an integer, a smooth and a black-box formulation. Explain the solver and profile choice for three of them. |
| A2 | Connect an agent: [runner setup](https://github.com/dietmarwo/optimization-jobs/blob/main/README.md), [MCP guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp), [chapter setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/SETUP.md). | Build and register a worker and connect the MCP client as documented. `describe_node({})` and `describe_worker({"worker":"lp-ip"})` must return the node and the worker description, including runnable examples. |
| A3 | **Capstone:** [LP/IP chapter](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/lp-ip/README.md) and its [prompt](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/lp-ip/prompt.md). | After the shared `build --chapter lp-ip`, run `python tutorials/agentic/run.py run --sources .. --bin-dir ../prompt-bin --output out/agent-lp --chapter lp-ip --cores 2` in optimization-jobs. Inspect the fractional bound, the integer enumeration and `lp-ip-summary.json`. |
| A4 | Interpret evidence: [result contract](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#termination-claim-and-evidence), [comparison with other systems](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/optimization-systems-for-ai-agents.md). | In optimization-jobs, `python tutorials/agentic/test_contract.py` passes the acceptance tests. Explain how the capstone's checked optimum differs from an infeasibility claim that has no exported certificate. A candidate from a run that stopped at a budget is not an optimum. |
| A5 | Compose work beyond one job: [campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/campaigns.md), [hydro walkthrough](https://github.com/dietmarwo/hydro-craft/blob/main/docs/command-jobs.md), [oscillator protocol](../tutorials/oscillator-topology-search/AGENT_PROTOCOL.md), [route search](../tutorials/gtoc1-route-search/README.md). | Run the smoke campaign of the walkthrough's section 5 from a shell. Inspect its dependent check, and explain why the MCP server refuses these command jobs. |

A Python objective can start at [P2](#python-user); a new native registered
model starts at [R5](#rust-user). Mathematicians and domain researchers also
have the [mathematicians guide](https://github.com/dietmarwo/experiment-craft/blob/main/docs/mathematicians-guide.md).

## Capstone checks

| Track | Capstone | Checked by |
| --- | --- | --- |
| Python | Hydro evaluation, optimization and campaign; the thermo port runs through the same driver | [Domain-evaluator workflow](https://github.com/dietmarwo/optimization-jobs/blob/main/.github/workflows/domain-evaluators.yml) |
| Rust | Native job API tutorial with the registered `curve-fit` application model | [fcmaes-rust CI](https://github.com/dietmarwo/fcmaes-rust/blob/main/.github/workflows/ci.yml) |
| Agent | LP/IP prompt chapter with its real worker and an independent enumeration | [Agentic tutorial CI](https://github.com/dietmarwo/optimization-jobs/blob/main/.github/workflows/agentic-tutorials.yml) |

In review, each capstone ran in under half a minute once its builds existed.
Installation and builds take longer and depend on the host.

## Limits and maintenance

- The capstones use synthetic or bounded models. Passing their checks does not
  establish physical, biological or operational validity.
- The hydro and thermo wheels are installed from their repositories, as the
  walkthrough's setup shows; they are not on PyPI.
- The tracks were walked through by agents in prepared environments, not yet
  by human readers.

`python scripts/check_learning_paths.py --sources ..` checks every link on
this page against sibling checkouts. With `--remote` it reads the other
repositories' current default branches instead; the workflow uses that mode on
pull requests and weekly. A missing file, a missing anchor, a URL that names
another branch than the default and a moved-page stub all fail. The page is
limited to 250 lines and names no release numbers.
