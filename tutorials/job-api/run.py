#!/usr/bin/env python3
"""Runnable native job tutorial. Build release binaries before running."""
import argparse, json, math, subprocess, tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORKER='fcmaes'
BINARY='fcmaes-job-worker'
EXAMPLES=['sphere', 'rosenbrock', 'cassini1', 'rosenbrock-cmaes', 'zdt1-mode', 'binh-korn-mode',
          'rastrigin-map-elites']

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);args=p.parse_args()
root=(args.root or Path(tempfile.mkdtemp(prefix=WORKER+'-jobs-'))).resolve()
runner=REPO/'target/release/optimization-run'
def call(*args):
    run=subprocess.run([str(runner),'--root',str(root),*map(str,args)],capture_output=True,text=True,check=True)
    return json.loads(run.stdout)

def dominated(a,b):
    """True if objective vector b dominates a."""
    return all(y<=x for x,y in zip(a,b)) and a!=b

def check_retry(spec,record,outcome):
    payload=spec['payload'];limit=payload['evaluations_per_retry']
    if payload['optimizer']['kind']=='cmaes':limit+=payload['optimizer']['parameters']['population']-1
    assert outcome['claim']=='best_found',outcome
    assert record['work']['evaluations']<=limit,record

def check_mode(spec,record,outcome):
    assert outcome['claim']=='non_dominated_set' and outcome['x'] is None,outcome
    details=outcome['details'];front=details['front'];problem=spec['payload']['problem']['id']
    assert front and details['hypervolume']['exact'],details
    for point in front:
        x,f=point['x'],point['objectives']
        assert not any(dominated(f,other['objectives']) for other in front),point
        if problem=='benchmark.zdt1':
            g=1+9*sum(x[1:])/(len(x)-1)
            assert math.isclose(f[1],g*(1-math.sqrt(x[0]/g)),rel_tol=1e-12,abs_tol=1e-12),point
            # Nothing can lie below the true front f2 = 1 - sqrt(f1).
            assert f[1]>=1-math.sqrt(f[0])-1e-12,point
        else:
            assert all(c<=0 for c in point['constraints']),point
    if problem=='benchmark.zdt1':
        # The true front's hypervolume for reference (1.1, 1.1) is 0.8767.
        assert details['hypervolume']['value']>0.85,details['hypervolume']

def check_map_elites(spec,record,outcome):
    assert outcome['claim']=='elite_archive',outcome
    details=outcome['details'];elites=details['elites'];shift=.4*5.12
    assert details['coverage']>.5,details['coverage']
    for elite in elites:
        x=elite['x'];z=[v-shift for v in x];half=len(x)//2
        fitness=10*len(x)+sum(v*v-10*math.cos(2*math.pi*v) for v in z)
        assert math.isclose(elite['fitness'],fitness,rel_tol=1e-9,abs_tol=1e-9),elite
        assert math.isclose(elite['descriptor'][0],sum(x[:half]),abs_tol=1e-9),elite
        assert math.isclose(elite['descriptor'][1],sum(x[half:]),abs_tol=1e-9),elite
    assert outcome['objective']==min(e['fitness'] for e in elites)

if not (root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id',WORKER,'--executable',REPO/'target/release'/BINARY)
report=[]
for example in EXAMPLES:
    spec_path=HERE/(example+'.json');spec=json.loads(spec_path.read_text())
    call('validate','--spec',spec_path)
    job=call('run','--spec',spec_path);result=call('result',job['directory'])
    assert job['state']=='completed',result
    assert result['runner']['accounting_complete'],result
    for record in result['results']:
        outcome=record['outcome']
        if WORKER=='lp-ip':
            if example=='infeasible':assert outcome['termination']=='infeasible' and outcome['claim']=='none'
            else:assert outcome['claim']=='numerical_optimum' and math.isclose(outcome['objective'],7.,abs_tol=1e-7)
        elif WORKER=='sqp':
            assert outcome['claim']=='first_order',outcome
            assert math.isclose(outcome['objective'],.25 if example=='simplex' else 0.,abs_tol=1e-6),outcome
        else:
            {'fcmaes.retry/1':check_retry,'fcmaes.mode/1':check_mode,
             'fcmaes.map-elites/1':check_map_elites}[spec['payload_schema']](spec,record,outcome)
    if example=='rosenbrock-cmaes':
        assert min(r['outcome']['objective'] for r in result['results'])<1e-10
    first=result['results'][0]['outcome']
    summary={k:v for k,v in first['details'].items() if k not in ('front','elites')}
    report.append({'example':example,'profile':spec['payload_schema'],'job_id':job['id'],'directory':job['directory'],
                   'state':job['state'],'completed_tasks':result['completed_tasks'],
                   'first_outcome':{**first,'details':summary},'work':result['work']})
output={'node':str(root),'worker':WORKER,'runs':report}
(root/'tutorial-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
