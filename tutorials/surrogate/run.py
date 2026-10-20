#!/usr/bin/env python3
"""Run Branin, warm continuation, and a Rust RK4 simulation through the job API."""
import argparse, copy, hashlib, json, math, os, subprocess, tempfile
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path)
p.add_argument('--bin-dir',type=Path,default=Path(os.environ.get('CARGO_TARGET_DIR',REPO/'target'))/'release')
args=p.parse_args();root=(args.root or Path(tempfile.mkdtemp(prefix='surrogate-tutorial-'))).resolve();root.mkdir(parents=True,exist_ok=True)
bins=args.bin_dir.resolve();runner=bins/'optimization-run'
for binary in [runner,bins/'fcmaes-job-worker',bins/'examples/simulation_worker']:
    if not binary.is_file():p.error(f'missing {binary}; build the release runner, worker and simulation_worker example')
def call(*argv):
    r=subprocess.run([str(runner),'--root',str(root),*map(str,argv)],capture_output=True,text=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr)
    return json.loads(r.stdout)
def run(spec,path):
    path.write_text(json.dumps(spec,indent=2)+'\n');call('validate','--spec',path)
    job=call('run','--spec',path);assert job['state']=='completed',job
    result=call('result',job['directory']);assert result['runner']['accounting_complete'],result
    assert result['work']['evaluations']==spec['payload']['evaluations'],result
    o=result['results'][0]['outcome'];assert o['claim']=='best_found' and o['evidence']['level']=='rechecked',o
    return job,result
if not(root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id','fcmaes','--executable',bins/'fcmaes-job-worker')
call('workers','add','--id','simulation','--executable',bins/'examples/simulation_worker')
spec=json.loads((HERE/'branin.json').read_text());job,result=run(spec,root/'branin.json');x=result['results'][0]['outcome']['x']
value=(x[1]-5.1*x[0]*x[0]/(4*math.pi**2)+5*x[0]/math.pi-6)**2+10*(1-1/(8*math.pi))*math.cos(x[0])+10
assert math.isclose(value,result['results'][0]['outcome']['objective'],abs_tol=1e-10)
# The worker declares its log and snapshot as artifacts: the result indexes
# them with their SHA-256, and jobs artifact (MCP get_artifact) reads them.
index=result['runner']['artifact_index'];snapshot=index['surrogate-observations-0']
log=call('jobs','artifact',job['id'],'surrogate-0','--offset',0,'--limit',1000)
assert sum(r['record']=='event' and r['data']['event']=='completed' for r in log['records'])==48,log['available']
data=call('jobs','artifact',job['id'],'surrogate-observations-0')['content'].encode()
assert hashlib.sha256(data).hexdigest()==snapshot['sha256'];(root/'previous.json').write_bytes(data)
warm=copy.deepcopy(spec);warm['request_key']='surrogate-tutorial-warm-v1';warm['payload']['evaluations']=8
warm['payload']['warm_start']={'input':'previous'}
warm['inputs']=[{'name':'previous','file':'previous.json','sha256':snapshot['sha256']}]
warm_job,warm_result=run(warm,root/'warm.json');assert warm_result['results'][0]['outcome']['details']['warm_start_points']==48
sim_job,sim_result=run(json.loads((HERE/'oscillator.json').read_text()),root/'oscillator.json')
report={'node':str(root),'branin':{'job':job,'result':result},'warm':{'job':warm_job,'result':warm_result},'simulation':{'job':sim_job,'result':sim_result}}
(root/'tutorial-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'report':str(root/'tutorial-report.json'),'branin_best':result['results'][0]['outcome']['objective'],
    'warm_new_evaluations':warm_result['work']['evaluations'],'simulation_best':sim_result['results'][0]['outcome']['objective']},indent=2))
