#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
public_root="$(cd -- "$script_dir/../.." && pwd)"
external_root="${1:-$public_root/../fcmaes-optimizer-bench}"
sources="$script_dir/sources"

mkdir -p "$external_root"
sed "s|__FCMAES_CORE__|$public_root/crates/fcmaes-core|g" \
    "$sources/Cargo.toml.reference" > "$external_root/Cargo.toml"

for package in \
    common \
    fcmaes-adapter \
    cmaes-adapter \
    genetic-algorithms-adapter \
    math-optimisation-adapter \
    argmin-adapter
do
    mkdir -p "$external_root/$package/src"
    sed "s|__FCMAES_CORE__|$public_root/crates/fcmaes-core|g" \
        "$sources/$package/Cargo.toml.reference" \
        > "$external_root/$package/Cargo.toml"
    cp "$sources/$package/src/"*.rs "$external_root/$package/src/"
done

cp "$public_root/examples/src/gtop.rs" "$external_root/common/src/gtop.rs"

echo "$external_root"

