"""Real CLI cancel/deadline regression for a plain objective sleeping 30 ms per call.
Build optimization-run, fcmaes-job-worker and --example stop_worker first.
"""
import argparse, hashlib, json, pathlib, subprocess, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
node=pathlib.Path(tempfile.mkdtemp(prefix='fcmaes-stop-')); runner=ROOT/'target/release/optimization-run'
def call(*args):
 proc=subprocess.run([str(runner),'--root',str(node),*map(str,args)],capture_output=True,text=True,timeout=15,check=True)
 return json.loads(proc.stdout)
call('node','init','--capacity',2)
call('workers','add','--id','sleep','--executable',ROOT/'target/release/examples/stop_worker')
records=[]
for kind,batch in [('biteopt',None),('de',None),('biteopt',4),('de',4)]:
 for reason in ['cancel','deadline']:
  key=f'{kind}-{batch}-{reason}';parameters={}
  if batch:parameters['batch']=batch
  inner=2 if batch else 1
  spec=dict(schema_version='optimization.job/0.1',request_key=key,worker='sleep',payload_schema='fcmaes.retry/1',
   payload=dict(problem=dict(id='app.sleep',version='1',config=dict(dimension=3),parameters=dict(milliseconds=30),bounds=dict(lower=[-5.],upper=[5.],broadcast=True)),
    optimizer=dict(kind=kind,parameters=parameters),seed=3,retries=1,evaluations_per_retry=10000,evaluation_limit_mode='hard' if kind=='biteopt' else 'soft'),
   resources=dict(cpu_slots=inner,outer_workers=1,inner_threads=inner),limits=dict(wall_time_seconds=10. if reason=='cancel' else .5,grace_seconds=1.),inputs=[])
  path=node/f'{key}.json';path.write_text(json.dumps(spec));id=hashlib.sha256(key.encode()).hexdigest()[:32]
  proc=subprocess.Popen([str(runner),'--root',str(node),'run','--spec',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  began=time.perf_counter();cancel_at=None
  try:
   if reason=='cancel':
    until=time.monotonic()+10
    while time.monotonic()<until:
     job=node/'jobs'/id/'job.json'
     if job.exists() and json.loads(job.read_text()).get('state')=='running':break
     time.sleep(.005)
    else:raise TimeoutError('job never started')
    time.sleep(.2);cancel_at=time.perf_counter();call('jobs','cancel',id)
   stdout,stderr=proc.communicate(timeout=15)
   assert proc.returncode==0,(stdout,stderr)
   job=json.loads(stdout);run=json.loads((pathlib.Path(job['directory'])/'run.json').read_text())
   assert job['state']==('cancelled' if reason=='cancel' else 'timed_out'),job
   assert run['runner']['accounting_complete'],run
   tasks=run['results'];assert len(tasks)==1,run
   result=tasks[0];out=result['outcome'];work=result['work']
   assert out['termination']=='cancelled' and out['evidence']['level']=='claimed',out
   assert out['details']['recheck']=='skipped: cancelled'
   assert 0<work['evaluations']<10000 and work['validation_evaluations']==0
   assert work['cancelled_evaluations']==0
   elapsed=time.perf_counter()-began;latency=time.perf_counter()-cancel_at if cancel_at else None
   assert latency is None or latency<1.0,latency
   records.append(dict(optimizer=kind,batch=batch,reason=reason,state=job['state'],evaluations=work['evaluations'],validation_evaluations=work['validation_evaluations'],cancelled_evaluations=0,termination=out['termination'],evidence=out['evidence']['level'],seconds=elapsed,cancel_latency_seconds=latency))
   print(records[-1],flush=True)
  finally:
   if proc.poll() is None:
    try:call('jobs','cancel',id,'--now')
    finally:proc.kill();proc.wait()
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(records,indent=2)+'\n')
