#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
public_root="$(cd -- "$script_dir/../.." && pwd)"
external_root="${FCMAES_COMPARISON_WORKSPACE:-$public_root/../fcmaes-optimizer-bench}"
raw_dir="$script_dir/raw"
logs_dir="$script_dir/logs"

mkdir -p "$raw_dir" "$logs_dir"
"$script_dir/prepare_external.sh" "$external_root" >/dev/null

cargo build --release --workspace --manifest-path "$external_root/Cargo.toml"
cp "$external_root/Cargo.lock" "$script_dir/Cargo.lock.reference"

common_args=(
    --runs 100
    --workers 24
    --evaluations 240000
    --retries 24
    --evaluations-per-retry 10000
    --seed 1
    --resume
)

export RAYON_NUM_THREADS=24

"$external_root/target/release/fcmaes-benchmark-adapter" \
    --mode retry "${common_args[@]}" \
    --output "$raw_dir/fcmaes_biteopt_retry.tsv" \
    2> >(tee -a "$logs_dir/fcmaes_biteopt_retry.log" >&2)

"$external_root/target/release/fcmaes-benchmark-adapter" \
    --mode advanced "${common_args[@]}" \
    --output "$raw_dir/fcmaes_advanced_retry.tsv" \
    2> >(tee -a "$logs_dir/fcmaes_advanced_retry.log" >&2)

"$external_root/target/release/fcmaes-benchmark-adapter" \
    --mode batch "${common_args[@]}" \
    --output "$raw_dir/fcmaes_biteopt_batch.tsv" \
    2> >(tee -a "$logs_dir/fcmaes_biteopt_batch.log" >&2)

"$external_root/target/release/cmaes-benchmark-adapter" \
    --mode population "${common_args[@]}" \
    --output "$raw_dir/cmaes_population.tsv" \
    2> >(tee -a "$logs_dir/cmaes_population.log" >&2)

"$external_root/target/release/cmaes-benchmark-adapter" \
    --mode bipop "${common_args[@]}" \
    --output "$raw_dir/cmaes_bipop.tsv" \
    2> >(tee -a "$logs_dir/cmaes_bipop.log" >&2)

"$external_root/target/release/genetic-algorithms-benchmark-adapter" \
    "${common_args[@]}" \
    --output "$raw_dir/genetic_algorithms_lshade.tsv" \
    2> >(tee -a "$logs_dir/genetic_algorithms_lshade.log" >&2)

"$external_root/target/release/math-optimisation-benchmark-adapter" \
    "${common_args[@]}" \
    --output "$raw_dir/math_optimisation_de.tsv" \
    2> >(tee -a "$logs_dir/math_optimisation_de.log" >&2)

"$external_root/target/release/argmin-benchmark-adapter" \
    "${common_args[@]}" \
    --output "$raw_dir/argmin_pso.tsv" \
    2> >(tee -a "$logs_dir/argmin_pso.log" >&2)

python3 "$script_dir/render_report.py"
