"""Recompute all recorded medians from the bundled raw confirmation data."""
from pathlib import Path
import gzip,json,statistics
ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'docs/surrogate/advanced-evidence'
with gzip.open(EVIDENCE/'confirmation.jsonl.gz','rt') as f:records=[json.loads(line) for line in f]
summary=json.loads((EVIDENCE/'summary.json').read_text())
assert len(records)==summary['records']
assert not any('status' in r for r in records)
for row in summary['rows']:
 data=[r['best'] for r in records if r['variant']==row['variant'] and r['problem']==row['problem']]
 assert statistics.median(data)==row['median_loss']
print(json.dumps({'verified_records':len(records),'verified_rows':len(summary['rows'])}))
