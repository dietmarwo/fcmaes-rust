#!/usr/bin/env python3
"""Runnable native job tutorial. Build release binaries before running."""
import argparse, json, math, subprocess, tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORKER='fcmaes'
BINARY='fcmaes-job-worker'
EXAMPLES=['sphere', 'rosenbrock', 'cassini1']

p=argparse.ArgumentParser();p.add_argument('--root',type=Path);args=p.parse_args()
root=(args.root or Path(tempfile.mkdtemp(prefix=WORKER+'-jobs-'))).resolve()
runner=REPO/'target/release/optimization-run'
def call(*args):
    run=subprocess.run([str(runner),'--root',str(root),*map(str,args)],capture_output=True,text=True,check=True)
    return json.loads(run.stdout)
if not (root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id',WORKER,'--executable',REPO/'target/release'/BINARY)
report=[]
for example in EXAMPLES:
    spec=HERE/(example+'.json')
    call('validate','--spec',spec)
    job=call('run','--spec',spec);result=call('result',job['directory'])
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
            assert outcome['claim']=='best_found',outcome
            assert record['work']['evaluations']<=1000
    report.append({'example':example,'job_id':job['id'],'directory':job['directory'],'state':job['state'],'completed_tasks':result['completed_tasks'],'first_outcome':result['results'][0]['outcome'],'work':result['work']})
output={'node':str(root),'worker':WORKER,'runs':report}
(root/'tutorial-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
