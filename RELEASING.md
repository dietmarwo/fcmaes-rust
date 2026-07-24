# Publishing fcmaes-rust

The public repository produces two synchronized artifacts:

| Registry | Package | Version source |
|---|---|---|
| crates.io / docs.rs | `fcmaes-core` | `[workspace.package].version` |
| PyPI | `fcmaes-rust` | derived by Maturin from `fcmaes-py` |

The initial prepared version is 0.1.1. Publication is irreversible: never
reuse a version after uploading it to either registry.

## One-time registry setup

1. Create and protect a GitHub environment named `release`.
2. In PyPI, create a pending Trusted Publisher with:
   owner `dietmarwo`, repository `fcmaes-rust`, workflow
   `python-release.yml`, environment `release`, and project `fcmaes-rust`.
3. Confirm immediately before release that `fcmaes-core` and `fcmaes-rust`
   are still available.
4. The first `fcmaes-core` release must be published manually with a
   least-privilege crates.io token. After it exists, configure its Trusted
   Publisher for `publish-crates.yml` and environment `release`, then set the
   GitHub repository variable `CRATES_IO_TRUSTED_PUBLISHING` to `true`.

Do not set that repository variable before crates.io has accepted the Trusted
Publisher configuration. With the variable absent, the workflow performs all
checks and intentionally skips its upload steps.

## Pre-release checks

Run from the repository root:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo +1.88.0 check --workspace --locked
cargo doc -p fcmaes-core --no-deps
cargo package -p fcmaes-core --list
cargo publish -p fcmaes-core --dry-run --locked

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "maturin[patchelf]>=1.7,<2" numpy scipy pytest
env -u CONDA_PREFIX VIRTUAL_ENV="$PWD/.venv" \
  PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/maturin develop --release --locked
.venv/bin/python -m pytest
.venv/bin/maturin build --release --locked --compatibility pypi
.venv/bin/maturin sdist
```

Inspect the crate, wheel, sdist, README rendering and metadata. Install both
Python artifacts in clean environments outside the checkout and run
`scripts/smoke_python_package.py`. Compile the crate README example against
the packaged crate from a separate Cargo project.

Maturin removes unrelated workspace members from the sdist manifest. Its
standalone `sdist` command therefore normalizes the included lockfile when the
archive is built by pip; unlike wheel and Cargo package builds, it does not
offer a `--locked` option. The clean sdist installation below is the required
completeness and dependency-resolution check.

The release commit must have:

- a clean working tree;
- the intended version in `Cargo.toml` and `Cargo.lock`;
- a dated `CHANGELOG.md` entry;
- passing CI;
- no uncommitted generated artifacts.

## First crates.io release

Authenticate without placing the token in shell history:

```bash
cargo login
cargo publish -p fcmaes-core --locked
```

Verify the crate page, compile `cargo add fcmaes-core` in a clean project, and
check the docs.rs build. Then configure crates.io Trusted Publishing and the
repository variable described above.

## TestPyPI

Build the same artifacts from the intended release commit and upload them to
TestPyPI using a short-lived test credential or a dedicated trusted workflow.
Install with the production index available only for dependencies:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  fcmaes-rust==0.1.1
```

If testing requires changing an artifact, increment the version. Registry
files cannot be replaced.

## Tag and publish

After the manual first crates.io upload and successful TestPyPI check:

```bash
git tag -a v0.1.1 -m "Release fcmaes-rust 0.1.1"
git push origin v0.1.1
```

The tag must exactly equal `v` plus the Cargo package version. The Python
workflow builds, installs and smoke-tests every promised wheel family, tests
the source distribution, attests the artifacts and publishes through PyPI
Trusted Publishing. The crates workflow validates the package; for later
versions it also publishes through crates.io Trusted Publishing.

Create the GitHub release only after both registry pages and docs.rs have been
verified.

## Post-release verification

In clean directories:

```bash
cargo new fcmaes-registry-test
cd fcmaes-registry-test
cargo add fcmaes-core
cargo run --release

python -m venv fcmaes-wheel-test
fcmaes-wheel-test/bin/python -m pip install fcmaes-rust
fcmaes-wheel-test/bin/python -c \
  'import fcmaes_rust; print(fcmaes_rust.__version__); print(fcmaes_rust.phase1_build_info())'
```

Confirm the README, license, project links, wheel matrix, provenance and
release notes on crates.io, docs.rs, PyPI and GitHub.
