"""Recompute the recorded advanced-surrogate summaries from their raw data.

The delivered confirmation (seeds 201-230): record count, medians exactly
and paired geometric ratios to 1e-12 relative. Its bootstrap procedure was
not recorded, so its intervals are not recomputed. The trust-region review
confirmation (seeds 301-330): the whole summary, intervals included,
exactly, with summarize_advanced_panel.py's recorded bootstrap seed. The
handoff panels: the development selection and every confirmation summary,
exactly.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/surrogate/advanced-evidence"
spec = importlib.util.spec_from_file_location(
    "summarize", Path(__file__).with_name("summarize_advanced_panel.py")
)
summarize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize)
spec = importlib.util.spec_from_file_location(
    "select", Path(__file__).with_name("select_handoff_fraction.py")
)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def rows(summary):
    return {(r["problem"], r["variant"]): r for r in summary["rows"]}


delivered = json.loads((EVIDENCE / "summary.json").read_text())
records = summarize.load(EVIDENCE / "confirmation.jsonl.gz")
assert len(records) == delivered["records"] and not any("status" in r for r in records)
recomputed = rows(summarize.summary(records, bootstrap=1, seed=0))
for key, row in rows(delivered).items():
    assert recomputed[key]["median_loss"] == row["median_loss"], key
    assert recomputed[key]["median_seconds"] == row["median_seconds"], key
    ratio = recomputed[key]["geometric_ratio"]
    assert abs(ratio - row["geometric_ratio"]) <= 1e-12 * row["geometric_ratio"], key

review_dir = EVIDENCE / "trust-region-review"
review = json.loads((review_dir / "summary.json").read_text())
manifest = json.loads((review_dir / "confirmation-manifest.json").read_text())
provenance = json.loads((review_dir / "provenance.json").read_text())
assert manifest["seeds"] == list(range(301, 331))
assert manifest["binary_sha256"] == provenance["binary_sha256"]
review_records = summarize.load(review_dir / "confirmation.jsonl.gz")
assert len(review_records) == 600 and not any("status" in r for r in review_records)
assert {r["seed"] for r in review_records} == set(manifest["seeds"])
bootstrap = review["bootstrap"]
assert (
    summarize.summary(review_records, bootstrap["resamples"], bootstrap["seed"])
    == review
)

handoff = EVIDENCE / "handoff"
development = [handoff / f"development-{b}.jsonl.gz" for b in (160, 480)]
assert selection.select(development) == json.loads(
    (handoff / "development-selection.json").read_text()
)
summaries = 0
for path in sorted(handoff.glob("confirmation-*-vs-*.json")):
    budget, reference = path.stem.removeprefix("confirmation-").split("-vs-")
    recorded = json.loads(path.read_text())
    panel = summarize.load(handoff / f"confirmation-{budget}.jsonl.gz")
    manifest = json.loads(
        (handoff / f"confirmation-{budget}-manifest.json").read_text()
    )
    assert manifest["seeds"] == list(range(401, 431)) and manifest["budget"] == int(
        budget
    )
    assert not any("status" in r for r in panel)
    b = recorded["bootstrap"]
    assert summarize.summary(panel, b["resamples"], b["seed"], reference) == recorded, (
        path
    )
    summaries += 1
assert summaries == 8

print(
    json.dumps(
        {
            "handoff": {"confirmation_summaries": summaries},
            "delivered": {"records": len(records), "rows": len(delivered["rows"])},
            "trust_region_review": {
                "records": len(review_records),
                "rows": len(review["rows"]),
            },
        }
    )
)
