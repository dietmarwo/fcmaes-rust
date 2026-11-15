# Learning paths

Choose a reader track inside the [repository overview](repositories.md).
For a problem already in hand, use the
[solver-selection guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md);
for agent discovery, that guide remains the entry point.
These tracks index existing lessons. Install prerequisites from each linked
setup page, then run its checkpoint from the named repository root.
Use a new output directory for every run. `out/` below is a reader-created
results directory, not checked-in evidence.

## Python user

Start here with a Python objective or data to fit. No Rust programming is
required; the command-job capstone builds existing Rust executables on Linux.

| Stage | Goal and material | Checkpoint | Measured stage time |
| --- | --- | --- | --- |
| P1 | Minimize a bounded objective with [parallel retry](https://github.com/dietmarwo/fast-cma-es/blob/master/ai-context.md) and the [Python tutorials](https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/README.adoc). | After installation, `python -c "from fcmaes import retry; from scipy.optimize import Bounds; r=retry.minimize(lambda x: sum(v*v for v in x), Bounds([-5]*3,[5]*3), num_retries=8, max_evaluations=2000); print(r.fun,r.nfev)"` prints the best value and actual work; workers default to the available CPU count. | Not measured |
| P2 | Score a population natively: [hydro tutorial](https://github.com/dietmarwo/hydro-craft/blob/main/docs/tutorial.md) and [examples](https://github.com/dietmarwo/hydro-craft/blob/main/examples/python/README.md). | In hydro-craft, `python examples/python/fcmaes_cmaes_population.py` prints a fit and held-out metrics; inspect its graded penalty for invalid rows. | 0.737 s execution |
| P3 | Use linear/integer or smooth constrained structure: [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md). | In optimization-jobs, run `python tutorials/structured/run.py --chapter LP1 --output out/lp1`, then the same command with `--chapter SQP1 --output out/sqp1`; both summaries contain independent checks. | Not measured |
| P4 | Calibrate a stochastic simulator: [CRN guide](https://github.com/dietmarwo/CRN-craft/blob/main/docs/guide/01-install.md), [C1 lesson](https://github.com/dietmarwo/CRN-craft/blob/main/tutorials/optimization/C1-calibration.md), [simulator setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/simulators/README.md). | In CRN-craft, `python tutorials/optimization/run.py C1 --output out/c1` checks fresh/bound candidate-stream agreement and disjoint-seed held-out moments. | Not measured |
| P5 | **Capstone:** a repeatable [hydro command job](https://github.com/dietmarwo/hydro-craft/blob/main/docs/command-jobs.md), followed by the [domain campaign](https://github.com/dietmarwo/experiment-craft/blob/main/examples/campaigns/domain-evaluators/README.md). | After the walkthrough's setup and command-job replay, in hydro-craft run `python ../experiment-craft/examples/campaigns/domain-evaluators/run.py --sources .. --evaluator hydro-craft --preset smoke --seeds 1,2,3,4,5 --workers 2 --output out/hydro-five`; inspect numeric folds in `out/hydro-five/checks.json` and execution failures in `out/hydro-five/report/report.md`. | 21.257 s execution |

P3's [solver-choice exercise](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/solver-choice/README.md)
explains when model structure makes LP/IP or SQP the appropriate tool.
From P5, move to [agent stage A2](#agent-builder) to expose registered workers
over MCP. Command jobs themselves still require a shell.

## Rust user

Start here when you write the objective or simulator in Rust. Run Cargo with
the linked project's toolchain and locked dependencies.

| Stage | Goal and material | Checkpoint | Measured stage time |
| --- | --- | --- | --- |
| R1 | First objective: [getting started](getting-started.md), [one-shot and ask/tell](optimizers.md). | Run the bounded-sphere program from getting started in your Rust crate with `cargo run --release`; it prints value, evaluations and coordinates. Repeat using the ordered ask/evaluate/tell loop from the optimizer guide on the same bounds. | Not measured |
| R2 | Choose a method and budget: [chooser](choosing-an-optimizer.md), [foundations](../foundations/README.md), [retry](retry.md). | `cargo run --release --locked --manifest-path foundations/Cargo.toml -- --lesson 3 --workers 2` checks serial/parallel ordered retry equality; explain your method and work budget before changing them. | Not measured |
| R3 | Run an application: [tutorial index](../tutorials/README.md), starting with [Rapier trebuchet](../tutorials/rapier-trebuchet/README.md); then [GTOC1](../tutorials/gtoc1/README.md). | `cargo run --release --locked --manifest-path tutorials/rapier-trebuchet/Cargo.toml -- --mode simulate --x 3.5,80,4,2.5,-0.75,-0.05,2,2` prints one deterministic physical replay. Continue with the linked smoke and recorded-result commands. | Not measured |
| R4 | Solve structured models in Rust: [LP API](https://github.com/dietmarwo/LP-IP-craft/blob/main/docs/api.md), [LP example](https://github.com/dietmarwo/LP-IP-craft/blob/main/crates/native/examples/solve.rs), [SQP example](https://github.com/dietmarwo/SQP-craft/blob/main/crates/sqp/examples/solve.rs). | In LP-IP-craft, `cargo run --release --locked -p lp-ip-craft-native --example solve` checks a binary model; in SQP-craft, `cargo run --release --locked -p sqp-craft-solver --example solve` checks a constrained solution and KKT residuals. | Not measured |
| R5 | **Capstone:** [register a model](job-api/models.md), then [run the native job tutorial](../tutorials/job-api/README.md); see [application workers](job-api/application-tutorials.md). | Build the runner, worker and `custom_worker` as shown; from fcmaes-rust run `export PATH="$PWD/target/release:$PATH"` before the model guide's bare runner commands; `python3 tutorials/job-api/run.py --root out/native-jobs` runs the registered curve-fit model and writes `tutorial-report.json` with independent checks. | 16.121 s native tutorial |

For mixed-solver formulations, return to [Python stage P3](#python-user).
For an agent using the worker from R5, continue at [A2](#agent-builder).

## Agent builder

Start with the shared guide. A model's discovery schema and result evidence
control what the agent can submit and claim.

| Stage | Goal and material | Checkpoint | Measured stage time |
| --- | --- | --- | --- |
| A1 | Choose from a problem card: [agent entry point](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md), [solver-choice lesson](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/solver-choice/README.md). | In optimization-jobs after structured setup, `python tutorials/structured/run.py --chapter X1 --output out/choice` checks the LP, integer, smooth and black-box formulations. Explain solver/profile choices for three of them. | 1.023 s execution |
| A2 | Connect an agent: [runner setup](https://github.com/dietmarwo/optimization-jobs/blob/main/README.md), [MCP guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#agents-and-mcp), [chapter setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/SETUP.md). | Build/register the selected worker and connect the MCP client as documented; call `describe_node({})` and `describe_worker({"worker":"lp-ip"})`. Both must return node or worker discovery, including runnable examples. | 0.053 s MCP session |
| A3 | **Capstone:** [LP/IP chapter](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/lp-ip/README.md) and [prompt](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/agentic/lp-ip/prompt.md). | After the shared `build --chapter lp-ip`, run `python tutorials/agentic/run.py run --sources .. --bin-dir ../prompt-bin --output out/agent-lp --chapter lp-ip --cores 2`; inspect the fractional bound, integer enumeration and `lp-ip-summary.json`. | 2.661 s execution |
| A4 | Interpret evidence: [result contract](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/job-api.md#termination-claim-and-evidence), [system guide](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/optimization-systems-for-ai-agents.md). | In optimization-jobs, `python tutorials/agentic/test_contract.py` passes the acceptance tests; explain the capstone's numerical optimum versus an infeasibility claim without an exported certificate. A budget-limited candidate is not an optimum. | 0.091 s contract tests |
| A5 | Compose work beyond one job: [campaigns](https://github.com/dietmarwo/experiment-craft/blob/main/docs/campaigns.md), [hydro walkthrough](https://github.com/dietmarwo/hydro-craft/blob/main/docs/command-jobs.md), [oscillator protocol](../tutorials/oscillator-topology-search/AGENT_PROTOCOL.md), [route search](../tutorials/gtoc1-route-search/README.md). | From optimization-jobs, run `python ../experiment-craft/examples/campaigns/domain-evaluators/run.py --sources .. --evaluator hydro-craft --preset smoke --workers 2 --output out/agent-campaign`; inspect its dependent check and explain why MCP refuses these command jobs. | Not measured |

A Python objective can start at [P2](#python-user); a new native registered
model starts at [R5](#rust-user). Mathematical and domain researchers also
have the existing [mathematicians guide](https://github.com/dietmarwo/experiment-craft/blob/main/docs/mathematicians-guide.md).

## Capstone checks and timing

| Track | Runnable capstone | Continuous check |
| --- | --- | --- |
| Python | Hydro evaluation, optimization and campaign; thermo port through the same driver | [Domain-evaluator workflow](https://github.com/dietmarwo/optimization-jobs/blob/main/.github/workflows/domain-evaluators.yml) |
| Rust | Native job API tutorial, including registered `curve-fit` application model | [fcmaes-rust CI](https://github.com/dietmarwo/fcmaes-rust/blob/main/.github/workflows/ci.yml) |
| Agent | LP/IP prompt chapter with its real worker and independent enumeration | [Agentic tutorial CI](https://github.com/dietmarwo/optimization-jobs/blob/main/.github/workflows/agentic-tutorials.yml) |

The recorded values are fresh-agent process times in a prepared environment
on 15 November 2026. P5 totals the documented evaluation/replay/five-seed
commands; R5 is the native tutorial only, excluding the own-model build.
A2 measures the MCP session, excluding registration; A4 measures contract tests.
All exclude reading, setup and interpretation. Unexercised stages remain
**not measured**; no human reading-time estimate is implied.

## Maintenance and gaps

Run `python scripts/check_learning_paths.py --sources ..` in fcmaes-rust
against sibling checkouts. `--remote` instead fetches current default branches
into temporary checkouts; the PR/weekly workflow uses that mode. Missing
files, missing anchors, non-default-branch URLs and moved-page stubs fail.
The page is limited to 250 lines and avoids release numbers and chapter counts.

- Hydro and thermo wheels currently need the linked source setup; package
  installation stays in that walkthrough so it can change in one place.
- The Rust structured-solver examples exist; the cross-solver X1–X4 teaching
  sequence remains Python-based.
- A human reader walkthrough has not been measured. Any agent review times
  must be identified as agent navigation/execution, not human learning times.
- The capstones use synthetic or bounded models; passing their checks does
  not establish physical, biological or operational validation.
