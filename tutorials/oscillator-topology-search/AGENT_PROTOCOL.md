# Agent proposal protocol

The agent is a discrete hypothesis generator, not a simulator or optimizer.
The current adapter contract is `oscillator-topology-proposal-v4`.

## Request

Rust writes one JSON object to subprocess stdin:

- `proposal_attempt`;
- the exact grammar;
- the minimized objective description;
- evaluated canonical topology keys, holdout scores, dimensions and motif
  classes;
- rejected/duplicate keys; and
- an optional `repair_error`.

The held-out reference list and kinetic vectors are omitted. A
`no-motif-hint` live control should additionally omit motif-class labels.

## Response

The subprocess writes one JSON object:

```json
{
  "edges": [0, 0, 0, 2, 0, 0, 2, 2, 0],
  "input_tokens": 1200,
  "output_tokens": 35
}
```

Rust validates nine values in `{0,1,2}`, two to six active edges, and no
isolated gene. Invalid JSON or grammar output triggers one repair request. A
second malformed response rejects the proposal. A duplicate consumes a
proposal attempt but no inner objective evaluations.

The live adapter does not rely on prose JSON compliance. Its Anthropic path
forces one `propose_topology` tool whose compact input schema permits exactly
nine integers in `{0,1,2}`. Rust remains authoritative for active-edge and
connectivity rules.

The OpenAI-compatible path uses a protocol-v4 candidate menu. Before every
request, the adapter enumerates the 12,024 valid topologies and removes all
evaluated and rejected keys. It then builds a deterministic menu of at most 96
unseen candidates from three sources: round-robin one-edge mutations of up to
eight elites, underrepresented structural classes, and hash-ordered random
immigrants. Opaque candidate IDs are shuffled deterministically. The model's
strict `response_format` permits only one of those IDs, which the adapter maps
back to the existing nine-edge Rust response. Thus model repetition cannot
consume another proposal attempt. Rust still validates the result
defensively.

Command, configuration and network errors do not trigger a format-repair API
call. Three consecutive adapter failures open a persistent circuit breaker;
the manifest records the typed counters and bounded final diagnostic. Resume
does not clear that breaker. The checked-in live adapter provides `--check` to
validate its local configuration and credential environment without making an
API request.

Loopback OpenAI-compatible endpoints may omit `api_key_env`. Remote endpoints
and all Anthropic endpoints require a named secret environment variable. The
adapter never permits an unauthenticated non-loopback URL.

## Evidence policy

- `agents/mock_agent.py` tests transport only and contains no held-out
  reference.
- Mock results never appear in `results/publication/comparison.md`.
- With no command, the agent manifest is `status: "not-run"`.
- A live row requires a concrete provider/model, a positive maximum-token
  budget, provider-reported usage and a dedicated result directory.
- Random, evolutionary and agent arms receive the same accepted-candidate
  target and inner evaluation budget.
- Reference rows are never inserted into any proposal archive.
- Changing the menu or structured-output policy changes the versioned agent proposal
  policy in `run.json`; old agent archives are preserved but cannot be resumed
  under the new contract.
