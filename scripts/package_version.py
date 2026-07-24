"""Print the synchronized workspace/package version."""

from pathlib import Path
import tomllib


manifest = Path(__file__).resolve().parents[1] / "Cargo.toml"
with manifest.open("rb") as source:
    version = tomllib.load(source)["workspace"]["package"]["version"]
print(version)
