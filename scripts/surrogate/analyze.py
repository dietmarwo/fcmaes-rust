#!/usr/bin/env python3
"""Summarize the checked-in smoke data; optional standard matplotlib figure."""
import argparse, json, statistics
from pathlib import Path

P=argparse.ArgumentParser(description=__doc__)
P.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[2]/'docs/surrogate/validation')
P.add_argument('--plot',action='store_true');args=P.parse_args();directory=args.directory
parallel=json.loads((directory/'smoke-8-threads.json').read_text())
serial=json.loads((directory/'smoke-1-thread.json').read_text())
assert len(parallel['runs'])==40 and len(serial['runs'])==40
identity=lambda r:(r['problem'],r['method'],r['seed'])
serial_map={identity(r):r for r in serial['runs']}
for row in parallel['runs']:
    other=serial_map[identity(row)]
    assert row['history']==other['history'],identity(row)
    assert row['best']==other['best'] and row['evaluations']==other['evaluations']
lines=['# Surrogate v0.1 — measured smoke results','',
    'Five fixed seeds (7, 11, 19, 23, 41), Branin and Hartmann-3, 48 optimization evaluations per run, logical batch size four. '
    'All four methods share the initial centre/LHS evaluations. The CMA-ES population is four. These are small deterministic smoke problems, not a statistical claim about Rust simulations.', '',
    '| Problem | Method | Median final objective | Median regret | Median wall ms, 8 threads | Median wall ms, 1 thread |',
    '|---|---|---:|---:|---:|---:|']
for problem in ['branin','hartmann3']:
    for method in ['gp','random','cmaes','biteopt']:
        rows=[r for r in parallel['runs'] if r['problem']==problem and r['method']==method]
        reference=[serial_map[identity(r)] for r in rows]
        med=lambda key:statistics.median(r[key] for r in rows)
        lines.append(f'| {problem} | {method} | {med("best"):.10g} | {med("regret"):.5g} | {med("wall_seconds")*1000:.3f} | {statistics.median(r["wall_seconds"] for r in reference)*1000:.3f} |')
lines+=['','The complete observation histories and final objectives match exactly between one and eight installed Rayon threads for all 40 corresponding runs. '
    'The one-thread run keeps the same four-point logical batches and evaluates them serially; it is a library determinism check, not a valid four-concurrent-evaluation job reservation.', '',
    'Wall timings include proposal work and objective execution, with an in-memory event sink. They exclude worker fsync and process startup. The host is not dedicated; do not treat these figures as a portable latency guarantee. '
    'The test environment has an eight-CPU cgroup quota. Random/CMA-ES/BiteOpt are faster on these cheap functions; the surrogate uses more compute to reduce evaluation count.', '']
sleepy=directory/'sleepy-8-threads.json'
if sleepy.exists():
    data=json.loads(sleepy.read_text());lines+=['## Synthetic evaluation latency','',
        'Two seeds (7, 11), 10 ms sleep added to every objective call, same 48-evaluation budget and four-point batches. This is an artificial latency test, not a real 1–60 s simulation campaign.', '',
        '| Problem | Method | Median wall ms | Median final objective |','|---|---|---:|---:|']
    for problem in ['branin','hartmann3']:
        for method in ['gp','random','cmaes','biteopt']:
            rows=[r for r in data['runs'] if r['problem']==problem and r['method']==method]
            lines.append(f'| {problem} | {method} | {statistics.median(r["wall_seconds"] for r in rows)*1000:.3f} | {statistics.median(r["best"] for r in rows):.10g} |')
lines+=['','## Interpretation','',
    'The GP improves final objective quality on these two low-dimensional small-budget cases, at a higher proposal cost than the population baselines. '
    'This does not establish the crossover for larger dimensions, thousands of evaluations, noisy simulations, TuRBO or asynchronous scheduling. '
    'Those are the next gates in the revised plan. No significance tests, held-out simulation claim or 16-core performance claim are made.', '']
(directory/'REPORT.md').write_text('\n'.join(lines))
if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    optima={'branin':0.39788735772973816,'hartmann3':-3.86278214782076}
    for ax,problem in zip(axes,['branin','hartmann3']):
        for method in ['gp','random','cmaes','biteopt']:
            rows=[r for r in parallel['runs'] if r['problem']==problem and r['method']==method]
            curves=[]
            for row in rows:
                running=float('inf');curve=[]
                for observation in row['history']:
                    running=min(running,observation['value']['value'])
                    curve.append(max(running-optima[problem],1e-12))
                curves.append(curve)
            median=[statistics.median(c[i] for c in curves) for i in range(48)]
            ax.semilogy(range(1,49),median,label=method)
        ax.set(title=problem,xlabel='Optimization evaluations',ylabel='Median simple regret')
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Small-budget smoke test: 5 seeds, batch size 4 (not a general ranking)')
    figure_path=directory/'convergence.svg'
    fig.savefig(figure_path);plt.close(fig)
    figure_path.write_text('\n'.join(line.rstrip() for line in figure_path.read_text().splitlines())+'\n')
print('Wrote',directory/'REPORT.md')
