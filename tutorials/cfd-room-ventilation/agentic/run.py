#!/usr/bin/env python3
"""Run and audit the prompt-first CFD workshop over the existing native CLI."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

BOUNDS = [(0.12, .88), (.12, .45), (.12, .88), (.12, .45), (.25, 1.5),
          (.2, .8), (.15, .85), (.15, .65), (-1.4, 1.4)]
METRICS = ('exposure', 'maximum_receptor', 'fan_power', 'final_mass_fraction')
DECISIONS = ('inlet_y', 'inlet_width', 'outlet_y', 'outlet_width', 'inlet_velocity',
             'baffle_x', 'baffle_y', 'baffle_length', 'baffle_angle')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def scalar(row):
    return float(row['exposure']) + .5*float(row['maximum_receptor']) + .2*float(row['fan_power']) + .5*float(row['final_mass_fraction'])


def selected(path):
    points = rows(path)
    chosen = [r for r in points if r['selected'] == '1']
    if len(chosen) != 1:
        raise ValueError('Exactly one selected design is required')
    for row in points:
        values = [float(row[k]) for k in METRICS]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite objective')
        if not math.isclose(scalar(row), float(row['quality']), rel_tol=0, abs_tol=1e-8):
            raise ValueError('Exported quality disagrees with its component metrics')
        for name, (lo, hi) in zip(DECISIONS, BOUNDS):
            if not lo <= float(row[f'decision_{name}']) <= hi:
                raise ValueError('Decision outside its declared bounds')
    return points, chosen[0]


def check_front(points):
    values = [[float(r[k]) for k in METRICS] for r in points]
    for i, x in enumerate(values):
        for j, y in enumerate(values):
            if i != j and all(a <= b for a, b in zip(y, x)) and any(a < b for a, b in zip(y, x)):
                raise ValueError('Dominated row in the retained four-objective MODE front')


def constraints(text, prefix):
    match = re.search(r'^' + prefix + r'_CONSTRAINTS (\[.*\])$', text, re.M)
    if not match:
        raise ValueError('Missing constraints for ' + prefix)
    values = json.loads(match[1])
    if len(values) != 4 or not all(math.isfinite(x) and x <= 0 for x in values):
        raise ValueError('Selected design violates a training/held-out constraint')
    return values


def run(args):
    output = args.output.resolve(); bins = args.bin_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    commands = []
    def command(name, argv):
        started = time.monotonic()
        try:
            result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            write(output/'failure.json', {'command': list(map(str, argv)), 'reason': 'process timeout'})
            raise RuntimeError('Native process exceeded the declared time budget') from exc
        log = output/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
        log.write_text(result.stdout + result.stderr)
        commands.append({'name': name, 'argv': list(map(str, argv)), 'exit_code': result.returncode,
                         'seconds': time.monotonic()-started, 'log': str(log.relative_to(output))})
        write(output/'commands.json', commands)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
        print(name + ': completed', flush=True)
        return result.stdout
    records = []
    for seed in args.seeds:
        for method, mode, filename, prefix in [('mode', 'multi', 'pareto.csv', 'MO'), ('qd', 'qd', 'archive.csv', 'QD')]:
            directory = output/f'{method}-seed-{seed}'
            text = command(f'{method}-{seed}', [bins/'cfd-room-ventilation', '--mode', mode,
                '--workers', args.workers, '--mo-evaluations', args.evaluations, '--popsize', 64,
                '--qd-evaluations', args.evaluations, '--qd-capacity', 100, '--qd-chunk-size', 64,
                '--flow-steps', 800, '--scalar-steps', 600, '--seed', seed, '--output', directory])
            points, chosen = selected(directory/filename)
            held_rows = rows(directory/'validation.csv')
            assert len(held_rows) == 1
            held = held_rows[0]
            assert int(held['source_count']) == 3
            assert all(held[f'decision_{name}'] == chosen[f'decision_{name}'] for name in DECISIONS)
            assert math.isclose(scalar(held), float(held['quality']), rel_tol=0, abs_tol=1e-8)
            training_constraints = constraints(text, prefix+'_SELECTED')
            held_constraints = constraints(text, prefix+'_VALIDATION')
            if method == 'mode':
                check_front(points)
            actual = int(rows(directory/'convergence.csv')[-1]['evaluations'])
            assert actual == args.evaluations
            records.append({'method': method, 'seed': seed, 'requested_evaluations': args.evaluations,
                            'actual_search_evaluations': actual, 'retained_points': len(points),
                            'training': chosen, 'held_out': held,
                            'training_constraints': training_constraints, 'held_out_constraints': held_constraints})
    # Predeclare the first requested seed as the field/grid representative,
    # independently of its measured quality; do not choose a winner afterwards.
    representative = args.seeds[0]
    command('verification', [bins/'verification', '--mode-results', output/f'mode-seed-{representative}/pareto.csv',
                             '--qd-results', output/f'qd-seed-{representative}/archive.csv', '--output', output/'verification'])
    grid = rows(output/'verification/resolution-study.csv')
    assert len(grid) == 18
    for record in records:
        if record['seed'] != representative:
            continue
        design = 'mode' if record['method'] == 'mode' else 'map_elites'
        for source, key in [('training', 'training'), ('held_out', 'held_out')]:
            point = next(r for r in grid if r['design'] == design and r['source_set'] == source and r['nx'] == '40')
            assert point['feasible'] == '1'
            for metric in ('quality', *METRICS):
                assert math.isclose(float(point[metric]), float(record[key][metric]), rel_tol=0, abs_tol=1e-8)
    channel = rows(output/'verification/channel-reference.csv')[0]
    assert float(channel['symmetry_relative_l2']) < 1e-10
    assert float(channel['maximum_transverse_velocity']) < 1e-3
    assert 1.45 < float(channel['maximum_to_mean_axial_velocity']) < 1.55
    assert float(channel['mass_imbalance']) < .01 and float(channel['residual']) < 1e-5
    aggregate = {}
    for method in ('mode', 'qd'):
        group = [r for r in records if r['method'] == method]
        aggregate[method] = {}
        for source in ('training', 'held_out'):
            values = [float(r[source]['quality']) for r in group]
            aggregate[method][source] = {'mean': statistics.mean(values),
                'sample_std': statistics.stdev(values) if len(values)>1 else None, 'values': values}
    summary = {'scope': 'Educational native CFD; no physical validation or solver ranking',
        'seeds': args.seeds, 'workers': args.workers, 'grid': [40, 24], 'flow_steps': 800, 'scalar_steps': 600,
        'process_timeout_seconds': args.timeout, 'representative_seed': representative,
        'total_search_evaluations': sum(r['actual_search_evaluations'] for r in records),
        'additional_work': 'Baseline, selected-design held-out replays and verification are outside search counts',
        'binary_sha256': {name: sha(bins/name) for name in ('cfd-room-ventilation','verification')},
        'records': records, 'aggregate': aggregate, 'resolution_study': grid, 'channel_reference': channel}
    write(output/'summary.json', summary)
    plot(output, summary)
    write(output/'checksums.json', {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'checksums.json'})


def plot(output, summary):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grid = summary['resolution_study']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, source in zip(axes, ('training', 'held_out')):
        baseline = float(next(r['quality'] for r in grid if r['design']=='baseline' and r['source_set']==source and r['nx']=='40'))
        ax.axhline(baseline, color='grey', linestyle='--', label='Baseline')
        for method, color in [('mode','#196b87'),('qd','#a56d27')]:
            values = summary['aggregate'][method][source]['values']
            ax.plot(summary['seeds'], values, marker='o', label=method.upper(), color=color)
        ax.set(title=source.replace('_',' ').title(), xlabel='Preselected optimizer seed', ylabel='Reporting quality (lower is better)')
        ax.set_xticks(summary['seeds']); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'fluid-seeds.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, source in zip(axes, ('training', 'held_out')):
        for design in ('baseline','mode','map_elites'):
            points = [r for r in grid if r['design']==design and r['source_set']==source]
            xs = [int(r['nx']) for r in points]; ys = [float(r['quality']) for r in points]
            ax.plot(xs,ys,marker='o',label=design)
            for x,y,r in zip(xs,ys,points):
                if r['feasible']!='1':ax.scatter([x],[y],marker='x',s=90,color='red',zorder=5)
        ax.set(title=source.replace('_',' ').title(),xlabel='Grid width (height = 0.6 x width)',ylabel='Reporting quality')
        ax.set_xticks([30,40,60]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'fluid-grid.png',dpi=150);plt.close(fig)
    data = []
    for design in ('baseline','mode','map_elites'):
        field = rows(output/'verification'/f'{design}-field.csv')
        assert len(field)==960
        u=np.array([float(r['u_lattice']) for r in field]).reshape(24,40)
        v=np.array([float(r['v_lattice']) for r in field]).reshape(24,40)
        c=np.array([float(r['concentration']) for r in field]).reshape(24,40)
        solid=np.array([int(r['solid']) for r in field]).reshape(24,40).astype(bool)
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(c))
        data.append((design,u,v,c,solid,float(field[0]['source_x_fraction'])*5,float(field[0]['source_y_fraction'])*3))
    vmax=max(float(np.hypot(d[1],d[2]).max()) for d in data)
    cmax=max(float(d[3].max()) for d in data)
    fig, axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for column,(design,u,v,c,solid,sx,sy) in enumerate(data):
        for row,(values,limit) in enumerate([(np.hypot(u,v),vmax),(c,cmax)]):
            ax=axes[row,column];masked=np.ma.array(values,mask=solid)
            mesh=ax.imshow(masked,origin='lower',extent=(0,5,0,3),vmin=0,vmax=limit,cmap='viridis')
            ax.set_facecolor('#343a40');ax.set(xlabel='x (m)',ylabel='y (m)',title=design.replace('_',' ').upper())
            if row==0:
                xx,yy=np.meshgrid((np.arange(40)+.5)*5/40,(np.arange(24)+.5)*3/24)
                ax.quiver(xx[::3,::3],yy[::3,::3],u[::3,::3],v[::3,::3],color='white',alpha=.6)
            else:ax.plot(sx,sy,'x',color='red',markersize=7)
            if column==2:fig.colorbar(mesh,ax=list(axes[row]),shrink=.8,label='Lattice speed' if row==0 else 'Final scalar concentration')
    fig.savefig(output/'fluid-fields.png',dpi=150);plt.close(fig)
    seed=summary['representative_seed'];front=rows(output/f'mode-seed-{seed}/pareto.csv');archive=rows(output/f'qd-seed-{seed}/archive.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    scatter=axes[0].scatter([float(r['fan_power']) for r in front],[float(r['exposure']) for r in front],c=[float(r['maximum_receptor']) for r in front],cmap='viridis')
    axes[0].set(xlabel='Normalized fan-power proxy',ylabel='Worst-source exposure',title='MODE: projection of four objectives')
    fig.colorbar(scatter,ax=axes[0],label='Maximum receptor metric')
    matrix=np.full((10,10),np.nan)
    for r in archive:matrix[int(r['grid_y']),int(r['grid_x'])]=float(r['quality'])
    mesh=axes[1].imshow(matrix,origin='lower',extent=(.09,2.025,0,1),aspect='auto',cmap='viridis_r')
    axes[1].set_facecolor('#dddddd');axes[1].set(xlabel='Flow proxy (m2/s per unit depth)',ylabel='Low-velocity occupied fraction',title='MAP-Elites: feasible behavior coverage')
    fig.colorbar(mesh,ax=axes[1],label='Reporting quality')
    fig.savefig(output/'fluid-alternatives.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--evaluations',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    if not 1<=args.workers<=16 or args.evaluations<128 or args.evaluations%64 or args.timeout<1:
        parser.error('workers 1..16, evaluations >=128 divisible by 64, positive timeout required')
    if len(set(args.seeds))!=len(args.seeds) or not all(0<=s<2**64 for s in args.seeds):
        parser.error('Use distinct unsigned 64-bit seeds')
    run(args)


if __name__=='__main__':main()
