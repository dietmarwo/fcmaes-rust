# Searching oscillator topologies with a split brain

This tutorial moves one level above
[fixed-topology Vilar oscillator design](../rebop-oscillator/README.md).
An outer process proposes *which gene regulates which*, while a native
numerical inner loop decides whether that signed network can actually produce
robust stochastic oscillations.

The architecture is a Rust port and protocol-focused extension of
[`autoresearch-circuit`](https://github.com/dietmarwo/autoresearch-circuit).
The new parts are a ReBop runtime model, fixed-evaluation inner optimization,
disjoint stochastic validation, held-out reference-label accounting, replayable
schema-v2 artifacts, and an explicit live-agent boundary. It is deliberately
a second split-brain tutorial: unlike variable-order GTOC1, the small outer
grammar has recognizable structural references.

![A discrete proposer is separated from the stochastic numerical evidence loop](images/architecture.svg)

## What the checked experiment found

The checked publication experiment compares three outer strategies under one
matched numerical protocol: seed 42, 200 accepted topologies, 16 independent
BiteOpt retries, 12,000 evaluations per retry, and 16 workers. Each accepted
topology therefore receives 192,000 requested objective calls, independent of
its 10–18-dimensional kinetic vector.

| Arm | Accepted / attempts | Exact references | Motif classes | Best | Median | Score < 1 |
|---|---:|---:|---:|---:|---:|---:|
| grammar-aware random | 200 / 210 | 0 | 5 | 0.613988 | 2.950889 | 10 |
| eight-elite evolutionary | 200 / 270 | 0 | 5 | 0.483957 | 2.636004 | 59 |
| Gemma 4 31B Q8, menu v4 | **200 / 200** | **repressilator at 188** | 5 | **0.471418** | **0.957210** | **103** |

Lower is better. The Gemma-plus-v4-menu policy's best score is 2.59% below
evolutionary and 23.22% below random. More importantly, its median and 103
sub-one results show that the outcome is not a single lucky incumbent. The v4
menu made every model response a novel accepted topology: zero duplicates,
invalid responses, or transport failures. The agent used 2,078,001 input and
3,200 output tokens through a local llama.cpp endpoint with thinking disabled.

The exact repressilator `000200220` was omitted from the held-out reference
list shown to the model and never entered proposal history. It was not absent
from the prompt: protocol v4 can select only an offered menu row. Deterministic
reconstruction shows that the unlabeled edge vector was offered on 17 attempts;
Gemma declined the first 16 and selected it at attempt 188, after declining six
consecutive elite-mutation offers on attempts 182–187. Neither offline control
exactly selected any of the four held-out encodings. This is feedback-guided
selection from an engineered candidate menu, not free generation. These are
descriptive results for one root seed and one model configuration, not a
general claim that an LLM dominates evolutionary search. See the complete
machine-generated [comparison](results/publication/comparison.md) and redacted
[agent provenance](results/publication/agent/provenance.json).

![Matched random, evolutionary, and Gemma outcomes](images/campaign-results.svg)

Summed per-topology optimizer wall times were 14,873 s for random, 14,835 s for
evolutionary, and 16,020 s for the agent. They were collected on separate
machines and exclude any unrecorded orchestration overhead, so they are
provenance—not a hardware comparison.

## The bounded topology grammar

There are three genes, one stochastic species per gene, and nine ordered edge
slots:

```text
A→A B→B C→C A→B A→C B→A B→C C→A C→B
```

Each slot is `0` absent, `1` activation, or `2` inhibition. A valid topology
has two to six active edges and no isolated gene. The slot vector is the
canonical identity, so duplicate proposals are rejected before they consume
an inner evaluation.

![The bounded grammar and canonical repressilator encoding](images/grammar-space.svg)

The structural archive key
`E{edges}-A{activations}-I{inhibitions}-S{self}-C{motifs}` is computed directly
from the decision. It is useful for search diagnostics, but it is not a
behavior descriptor.

## Runtime stochastic model

For target gene \(i\), the runtime production and degradation reactions are

\[
\varnothing \rightarrow X_i,\qquad
a_i + \sum_{j\rightarrow i} f_{ji}(X_j)
\]

\[
X_i \rightarrow \varnothing,\qquad \delta_i X_i .
\]

Activation and inhibition contributions are

\[
f^+_{ji}(x)=\alpha_{ji}\frac{x^{n_{ji}}}{K^{n_{ji}}+x^{n_{ji}}},
\qquad
f^-_{ji}(x)=\alpha_{ji}\frac{K^{n_{ji}}}{K^{n_{ji}}+x^{n_{ji}}},
\]

with \(K=20\) and initial copy count 10. This is the additive Hill model used
by `autoresearch-circuit`, not a multiplicative reinterpretation. Every
topology has exactly three species and six reactions; its active edges only
change the runtime propensity expressions.

![Hill contributions and the fixed six-reaction runtime skeleton](images/runtime-network.svg)

ReBop 0.9.7 declares `Rate::expr(Expr)` but does not export `Expr`, so an
external Rust crate cannot construct that public rate. The tutorial carries a
reduced MIT-licensed reimplementation of the ReBop 0.9.7 runtime expression and
Gillespie paths it needs. It is intentionally narrower than upstream and adds
a defensive invalid-propensity guard. A dual-source integration test requires
exact sparse mass-action replay against the released crate at three seeds;
runtime expressions are covered by analytic propensity and seeded replay
tests because upstream's private `Expr` cannot be named externally.
[DEPENDENCY_NOTICE.md](DEPENDENCY_NOTICE.md) gives the exact scope, numerical
delta and removal condition.

This three-species Hill network is **not** equivalent to the nine-species,
sixteen-reaction Vilar model. The fixed Vilar tutorial is a neighboring use
case, not a trajectory oracle for a different biological system.

## Variable inner optimization

Only active edges receive kinetic parameters:

| Block | Count | Optimizer representation | Physical bounds |
|---|---:|---|---|
| basal production | 3 | `log10` | 0.1–50 |
| degradation | 3 | `log10` | 0.005–1 |
| edge strength | active edges | `log10` | 0.1–100 |
| Hill coefficient | active edges | linear | 1–5 |

The dimension is therefore \(6+2|E|\), or 10–18 under this grammar. Every
candidate receives the same number of objective calls. That is auditable, but
not neutral: a six-edge network must search a larger space with the same
budget as a two-edge network. The CSV artifacts report both dimension and
requested/actual evaluations.

That asymmetry cuts against the best recorded arm: 176 of its 200 candidates
have dimension 16–18, compared with 123 for evolutionary and 90 for random.
Its median validation score is nevertheless lowest, and its median
training-to-validation gap is 0.571 versus 1.258 and 1.402. These are
descriptive one-seed measurements, not a complexity-corrected comparison.

BiteOpt minimizes the common-random-number training score. The final vector is
then evaluated on disjoint seeds. No simulator thread pool is enabled.

### Parallel inner retry

The split-brain boundary now has explicit numerical parallelism. For one
proposed topology, `--inner-retries R` creates `R` independent BiteOpt restarts
with a deterministic ordered seed schedule. `--workers W` runs at most
`min(R,W)` restarts concurrently; `--workers 0` uses available CPUs. Increasing
workers alone changes scheduling, not seeds, total evaluations or the selected
result. A regression test requires the same parameters and score with one and
two workers.

Both `R` and `W` default to the machine's physical-core count, capped by the
logical CPU quota visible to the process. On the 16-core/32-thread development
machine, the default is therefore 16 retries on 16 workers. Use
`--inner-retries 1 --workers 1` for a serial smoke or diagnostic run.
`--workers 0` deliberately opts into all visible logical CPUs but is
still capped by `R`.

`--evaluations E` is the budget **per retry**, so one topology requests
`R × E` objective evaluations. The candidate CSV records this total and the
actual total, while `run.json` records retries, requested workers and resolved
workers. For example, this uses 16 cores during each kinetic optimization:

```bash
cargo run --release --locked -- \
  --mode campaign --preset publication --strategy random \
  --accepted-candidates 200 --inner-retries 16 --workers 16 \
  --evaluations 1000 --seed 42 \
  --output results/local/random-r16-e1000
```

The regression test `parallel_inner_retries_are_worker_count_invariant` runs
the same seeded case with one and two workers and requires identical selected
parameters and score. It verifies scheduling invariance, not speedup; measure
throughput on the intended objective and host rather than extrapolating an
unarchived development timing.

The outer proposal loop remains sequential by design. After eight independent
bootstrap samples, the evolutionary arm selects uniformly among its eight
best validated topologies and applies one grammar-preserving edit. Every fifth
proposal is instead an independent random immigrant. The elite pool exploits
several promising neighborhoods, while the fixed 20% immigrant cadence
prevents exhaustion of the one-edit neighborhood around a single incumbent.
The agent likewise observes all prior validated results. Evaluating topology
batches from stale snapshots would define different outer algorithms.
Parallel retry fills the cores without weakening that feedback contract or
creating a nested simulator pool.

## Score contract

After 64 time units of burn-in, the model records 128 samples one unit apart.
For each gene, a fixed DFT plan over periods 8–64 measures:

- dominant period and target-period error;
- 10th-to-90th-percentile amplitude;
- spectral concentration around the dominant bin; and
- one- and two-period autocorrelation decay.

The aggregate additionally penalizes missing gene participation, period
disagreement, stochastic failure, replicate variation and molecular cost.
[`SCORE_SPEC.md`](SCORE_SPEC.md) freezes the formula.

Correctness does not circularly compare this scorer with another implementation
of the same formula. Tests use analytic sine, period-shift, flat-signal and
three-gene-participation fixtures, plus bit-identical seeded stochastic replay.

![Disjoint-seed traces for the three recorded incumbents](images/best-traces.svg)

## Structural references, not seeds

The four reference encodings are held out from proposal history:

- the three-inhibition repressilator is an oscillator reference;
- the `(+,+,−)` cycle is labelled **Goodwin-like** because this reduced
  three-species network is not Goodwin's original biochemical model;
- the positive cycle is a structural comparison, not guaranteed to oscillate;
  and
- mutual inhibition is a bistable toggle control, not an oscillator target.

The toggle adds `A→C` solely to keep gene C connected under the grammar.
Rediscovery means exact equality with the canonical vector; motif-class
discovery is reported separately.

![Held-out references and their signed edges](images/reference-motifs.svg)

See [MOTIFS.md](MOTIFS.md) for the encodings, classifier rules and primary
references.

## The agent boundary

An agent sees the grammar, previous topology keys, validation scores, motif
classifications and dimensions. It does not see the held-out reference list or
optimized kinetic vectors. It returns only:

```json
{"edges":[0,0,0,2,0,0,2,2,0],"input_tokens":123,"output_tokens":17}
```

Rust validates, canonicalizes and deduplicates the proposal. Invalid output
gets one bounded repair request. A duplicate consumes a proposal attempt but
no inner optimization budget. Provider usage is accumulated in `run.json`.
[`AGENT_PROTOCOL.md`](AGENT_PROTOCOL.md) defines the boundary and the
`no-motif-hint` control.

The mock emits only self-edge variants and contains none of the references. It
is a transport fixture:

```bash
cargo run --release --locked -- \
  --mode all --preset smoke \
  --agent-command 'python3 agents/mock_agent.py' \
  --output results/local/ci-smoke
```

For a real MiniMax Anthropic-compatible run, copy the example configuration,
set the secret locally, choose an explicit token limit, and record a separate
artifact directory:

```bash
osc_result_root="results/live-minimax-seed42"
mkdir -p "$osc_result_root"
cp config.live.example.json "$osc_result_root/agent-config.json"
export MINIMAX_API_KEY='...'

# Local-only check: validates the file, provider, model, token cap and key.
python3 agents/llm_agent.py \
  --config "$osc_result_root/agent-config.json" \
  --check

cargo run --release --locked -- \
  --mode campaign --preset publication --strategy agent \
  --accepted-candidates 20 \
  --agent-command "python3 agents/llm_agent.py --config $osc_result_root/agent-config.json" \
  --output "$osc_result_root"
```

The preflight makes no API request. After it passes, test one real proposal
before starting a long campaign. The Rust boundary distinguishes command or
network transport errors from malformed model responses; only the latter gets
the single repair request. Three consecutive adapter failures open a circuit
breaker, preserve the last diagnostic in `run.json`, and stop the arm. A
circuit-broken arm cannot silently resume: preserve or rename its directory,
fix and preflight the adapter, then start a fresh arm.

The MiniMax/Anthropic path uses the same transport proven by the GTOC1 route
search: both bearer and `X-Api-Key` authentication headers, SSE streaming and
adaptive-thinking filtering. Protocol v4 retains the single
`propose_topology` tool call. The provider may reason privately, but only the
tool's schema-constrained nine-edge input reaches Rust. The example caps the
response at 8,000 tokens; the previous 20,000-token unconstrained experiment
could exhaust its budget in thinking without returning a candidate.

### Local llama.cpp agent

The same adapter can call a local llama.cpp server without a fake API key.
For the 16 GB smoke-test GPU, the official 12B Gemma 4 Q4 conversion leaves
substantial room for a 32K context and KV cache:

```bash
llama-server \
  -hf ggml-org/gemma-4-12B-it-GGUF:Q4_0 \
  --alias gemma-4-12b-it-q4 \
  --gpu-layers all --ctx-size 32768 --flash-attn on \
  --host 127.0.0.1 --port 8080
```

In a second terminal:

```bash
osc_result_root="results/local/llamacpp-gemma4-12b-smoke"
mkdir -p "$osc_result_root"
cp config.llamacpp.example.json "$osc_result_root/agent-config.json"

python3 agents/llm_agent.py \
  --config "$osc_result_root/agent-config.json" \
  --check

printf '%s\n' \
  '{"proposal_attempt":1,"grammar":"nine digits in {0,1,2}; 2..=6 active; no isolated gene","objective":"minimize holdout oscillator score; seek distinct signed topologies","evaluated":[],"rejected_keys":[],"repair_error":null}' |
python3 agents/llm_agent.py \
  --config "$osc_result_root/agent-config.json"

cargo run --release --locked -- \
  --mode campaign --preset smoke --strategy agent \
  --accepted-candidates 1 --inner-retries 1 --workers 1 \
  --evaluations 200 --seed 42 \
  --agent-command "python3 agents/llm_agent.py --config $osc_result_root/agent-config.json" \
  --output "$osc_result_root"
```

The OpenAI-compatible request uses llama.cpp's schema-constrained
`response_format`, not model-specific function-call syntax. Protocol v4 first
removes every evaluated or rejected key from the 12,024-member grammar. It
then presents 96 unseen candidates: one third round-robin mutations of up to
eight elites, one third underrepresented structural classes, and one third
deterministic random immigrants. The model can return only one opaque menu ID,
which the adapter translates back to nine edges; Rust validates again
defensively. The IDs are shuffled, but each row explicitly identifies its
source. In the publication run, Gemma chose 167 elite mutations, seven
underrepresented-structure rows, and 26 random immigrants. A menu-matched
non-model selector has not yet been run, so score differences belong to the
complete menu–model policy and cannot be attributed to Gemma alone. The local URL
may omit `api_key_env` only because it is loopback; the adapter rejects an
unauthenticated remote endpoint. The example keeps the same 8,000-token ceiling
as MiniMax but disables Gemma's explicit thinking channel. In the 16 GB smoke
test, Gemma 4 12B ignored a request for at most 500 reasoning tokens, generated
more than 7,200 and hit the adapter timeout before returning nine integers.
Direct menu selection is therefore the honest local baseline.
It is a different proposer from thinking-enabled MiniMax and must remain a
separate experimental arm. With thinking disabled and the protocol-v3 complete
grammar schema, the 12B Q4 server returned a 27-token valid proposal in 0.73 s;
the end-to-end Rust smoke campaign accepted one topology on its first attempt.
After the v4 repair, a fresh two-candidate end-to-end smoke accepted both in
exactly two attempts. Each local request used about 5,300 prompt tokens and 16
output tokens and took about 3.1 seconds on the 16 GB test GPU; no duplicate or
invalid proposal was recorded.

Do not check in a local configuration or the key. A real agent row belongs in
the headline comparison only after its result directory records the concrete
provider/model configuration and `run.json` records provider token usage.

## From failed proposer to completed experiment

The first Gemma 4 31B run used protocol v3's complete grammar schema. Transport
and syntax worked, but the model accepted only 24 candidates in 2,500 attempts;
2,476 responses repeated an evaluated topology. Its best accepted score was
1.051480. This was a novelty-control failure, not an inner-budget or thinking
failure.

Protocol v4 moved novelty enforcement outside the model. Every request now
contains only unseen candidates, so the completed agent arm needed exactly 200
attempts for 200 accepted topologies. The old serial and failed-v3 publication
artifacts are not mixed with the headline evidence. The raw external result
directories may be retained as historical evidence, but `results/publication`
contains only the three matched completed arms.

## Reproduce and verify the publication evidence

Each arm contains its manifest, replay archive, tabular projection, convergence
history, and incumbent trace. The comparison command is read-only with respect
to those arm directories: it validates exact schema-v2 manifests and the full
numerical/proposal protocol, then rewrites only `comparison.md`.

```bash
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
python3 -m unittest agents/test_llm_agent.py

cargo run --release --locked -- \
  --mode report --preset publication --accepted-candidates 200 \
  --inner-retries 16 --workers 16 --evaluations 12000 \
  --seed 42 --output results/publication

python3 plot_results.py --check
```

To repeat an expensive arm, use a fresh output root and the campaign commands
above, advancing through checkpoints 20, 50, 100, 150, and 200. `--resume`
restores an arm only when schema version, strategy, seed, complete inner
protocol, resolved worker count, and proposal policy match. Changing any of
those boundaries fails before the archive is modified. Random, evolutionary,
and Gemma/MiniMax runs belong in separate directories until their manifests
have been checked; never splice partial agent archives or change models inside
one arm.

## Limitations

- The Hill model is a coarse regulatory abstraction; it does not model mRNA,
  promoters, binding species, growth or cell division.
- “Goodwin-like” describes a signed feedback core, not model equivalence.
- A toggle is expected to be bistable, so its oscillator score is a negative
  control rather than a success criterion.
- Two hundred proposals at one root seed support a matched case study, not a
  general algorithm ranking or uncertainty estimate. Under the exact
  rejection-sampled grammar, one specified three-edge reference has probability
  `1/2912` per independent valid random draw; the four references together have
  probability `1/728`, so 200 independent draws have
  `1-(727/728)^200 = 24.04%` probability of any exact hit before duplicate
  conditioning.
- The evolutionary arm uses eight elites and 20% random immigrants; other
  evolutionary policies could change the comparison.
- Inner retries parallelize one topology's numerical search. Topology proposals
  themselves remain sequential so evolutionary and agent decisions always use
  the latest validated archive.
- Agent conclusions apply to Gemma 4 31B Q8 with thinking disabled and a
  96-candidate v4 menu; model, quantization, prompt, or menu changes define a
  different strategy.
- No seeded uniform or fixed-rule selector has yet been run over the identical
  v4 menus. That ablation is required to separate candidate generation from
  model ranking.
- The compatibility copy should be removed once ReBop exports its runtime
  expression type upstream.

## Primary references

- D. T. Gillespie,
  [“Exact stochastic simulation of coupled chemical reactions”](https://doi.org/10.1021/j100540a008),
  *J. Phys. Chem.* 81, 2340–2361 (1977).
- M. B. Elowitz and S. Leibler,
  [“A synthetic oscillatory network of transcriptional regulators”](https://doi.org/10.1038/35002125),
  *Nature* 403, 335–338 (2000).
- B. C. Goodwin,
  [“Oscillatory behavior in enzymatic control processes”](https://doi.org/10.1016/0065-2571(65)90067-1),
  *Advances in Enzyme Regulation* 3, 425–437 (1965).
- T. S. Gardner, C. R. Cantor and J. J. Collins,
  [“Construction of a genetic toggle switch in Escherichia coli”](https://doi.org/10.1038/35002131),
  *Nature* 403, 339–342 (2000).
