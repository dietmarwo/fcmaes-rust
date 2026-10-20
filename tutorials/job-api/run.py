#!/usr/bin/env python3
"""Runnable native job tutorial. Build the release binaries and the custom_worker example first."""
import argparse, json, math, os, subprocess, sys, tempfile
from pathlib import Path
try:
    import jsonschema
except ImportError:
    jsonschema=None
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
WORKER='fcmaes'
BINARY='fcmaes-job-worker'
# An application worker with a model of its own (examples/custom_worker.rs).
APPLICATION=('curve-fit','examples/custom_worker')
EXAMPLES=['sphere', 'rosenbrock', 'cassini1', 'rosenbrock-cmaes', 'cassini1-bipop', 'rosenbrock-crfmnes',
          'zdt1-mode', 'binh-korn-mode', 'rastrigin-map-elites', 'sphere-cvt-cma-mae', 'binh-korn-evaluate',
          'exponential-fit']
sys.path.insert(0,str(REPO/'scripts/job-api'))
from shared_root import shared_root
# The schemas live in optimization-jobs, at the revision Cargo.toml pins.
SCHEMAS=shared_root()/'docs/job-api/schemas'
if jsonschema is None and os.environ.get('JOB_API_REQUIRE_SCHEMA')=='1':
    raise SystemExit('the jsonschema package is required (JOB_API_REQUIRE_SCHEMA=1)')

def conforms(schema,value):
    """Checks value against a published JSON Schema if jsonschema is installed."""
    if jsonschema is not None:
        jsonschema.validate(value,json.loads((SCHEMAS/schema).read_text()),cls=jsonschema.Draft202012Validator)

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
    payload=spec['payload'];limit=payload['evaluations_per_retry'];optimizer=payload['optimizer']
    if optimizer['kind'] in ('cmaes','crfmnes','pgpe'):
        # A soft limit: the last generation may overshoot by its population - 1.
        population=optimizer['parameters'].get('population',{'cmaes':31,'crfmnes':32,'pgpe':32}[optimizer['kind']])
        largest=optimizer['parameters'].get('max_population',512*population) if 'restarts' in optimizer['parameters'] else population
        limit+=largest-1
    if 'restarts' in optimizer['parameters']:
        assert outcome['details']['restarts']>=1 and record['work']['evaluations']>=payload['evaluations_per_retry'],outcome
    if payload['problem']['id']=='app.exponential-fit':
        check_fit(outcome)
    assert outcome['claim']=='best_found',outcome
    assert outcome['evidence']=={'level':'rechecked','checks':['objective','bounds']},outcome
    assert record['work']['evaluations']<=limit,record

def check_mode(spec,record,outcome):
    assert outcome['claim']=='non_dominated_set' and outcome['x'] is None,outcome
    assert outcome['evidence']['level']=='rechecked' and 'non_dominance' in outcome['evidence']['checks'],outcome
    details=outcome['details'];front=details['front'];problem=spec['payload']['problem']['id']
    assert details['front_source']=='final_population' and details['hypervolume']['normalization']=='none',details
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

def check_fit(outcome):
    """Recomputes the squared loss of a*exp(-b*t)+c on decay.csv (a=3, b=0.7, c=0.5 plus noise)."""
    rows=[tuple(map(float,line.split(','))) for line in (HERE/'decay.csv').read_text().split()[1:]]
    a,b,c=outcome['x']
    loss=sum((a*math.exp(-b*t)+c-y)**2 for t,y in rows)
    assert math.isclose(outcome['objective'],loss,rel_tol=1e-9),(outcome['objective'],loss)
    assert abs(a-3)<.05 and abs(b-.7)<.02 and abs(c-.5)<.02,outcome['x']

def check_map_elites(spec,record,outcome):
    assert outcome['claim']=='elite_archive',outcome
    details=outcome['details'];elites=details['elites'];shift=.4*5.12
    parameters=spec['payload']['optimizer']['parameters']
    budget=spec['payload']['generations']*parameters['chunk_size']
    if 'cma_generations' in parameters:
        budget+=parameters['cma_generations']*parameters['emitters']*parameters['population']
    assert record['work']['evaluations']==budget,record['work']
    assert details['coverage']>.5,details['coverage']
    rastrigin=spec['payload']['problem']['id']=='benchmark.rastrigin-projection'
    for elite in elites:
        x=elite['x'];z=[v-shift for v in x];half=len(x)//2
        if rastrigin:fitness=10*len(x)+sum(v*v-10*math.cos(2*math.pi*v) for v in z)
        else:fitness=sum(v*v for v in z)
        assert math.isclose(elite['fitness'],fitness,rel_tol=1e-9,abs_tol=1e-9),elite
        assert math.isclose(elite['descriptor'][0],sum(x[:half]),abs_tol=1e-9),elite
        assert math.isclose(elite['descriptor'][1],sum(x[half:]),abs_tol=1e-9),elite
    assert outcome['objective']==min(e['fitness'] for e in elites)
    assert outcome['evidence']=={'level':'rechecked','checks':['objective','bounds','niche']},outcome

def check_evaluate(spec,record,outcome):
    """Binh-Korn points: feasible exactly when both constraints are <= 0."""
    x=spec['payload']['points'][record['task_id']];a,b=x
    f=[4*a*a+4*b*b,(a-5)**2+(b-5)**2];c=[(a-5)**2+b*b-25,7.7-(a-8)**2-(b+3)**2]
    feasible=all(v<=0 for v in c)
    assert outcome['termination']=='evaluated' and outcome['x']==x,outcome
    assert outcome['claim']==('feasible' if feasible else 'none') and outcome['feasible']==feasible,outcome
    details=outcome['details']
    assert all(math.isclose(u,v,rel_tol=1e-12,abs_tol=1e-12) for u,v in zip(details['objectives']+details['constraints'],f+c)),details

if not (root/'node.json').exists():call('node','init','--capacity',8)
call('workers','add','--id',WORKER,'--executable',REPO/'target/release'/BINARY)
call('workers','add','--id',APPLICATION[0],'--executable',REPO/'target/release'/APPLICATION[1])
report=[]
for example in EXAMPLES:
    spec_path=HERE/(example+'.json');spec=json.loads(spec_path.read_text())
    conforms('optimization.job-0.1.schema.json',spec)
    call('validate','--spec',spec_path)
    job=call('run','--spec',spec_path);result=call('result',job['directory'])
    conforms('optimization.result-0.2.schema.json',result)
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
            {'fcmaes.retry/1':check_retry,'fcmaes.mode/1':check_mode,'fcmaes.map-elites/1':check_map_elites,
             'fcmaes.evaluate/1':check_evaluate}[spec['payload_schema']](spec,record,outcome)
    best=min((r['outcome']['objective'] for r in result['results'] if r['outcome']['objective'] is not None),default=None)
    if example=='rosenbrock-cmaes':assert best<1e-10,best
    if example=='rosenbrock-crfmnes':assert best<1e-20,best
    # The best known Cassini1 value is 4.9307 km/s.
    if example=='cassini1-bipop':assert best<5.4,best
    if example=='exponential-fit':
        resolved=call('validate','--spec',spec_path)['resolved']
        assert resolved['payload']['problem']['parameters']=={'loss':'squared'},resolved['payload']['problem']
    first=result['results'][0]['outcome']
    summary={k:v for k,v in first['details'].items() if k not in ('front','elites')}
    if example=='binh-korn-evaluate':
        assert [r['outcome']['claim'] for r in result['results']]==['feasible','feasible','none'],result
    report.append({'example':example,'profile':spec['payload_schema'],'job_id':job['id'],'directory':job['directory'],
                   'state':job['state'],'completed_tasks':result['completed_tasks'],
                   'first_outcome':{**first,'details':summary},'work':result['work']})
output={'node':str(root),'worker':WORKER,'runs':report}
(root/'tutorial-report.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
