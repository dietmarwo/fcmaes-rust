#!/usr/bin/env python3
"""Fixed-work paired native telemetry measurements. No scaling extrapolation."""
import argparse,csv,json,os,platform,random,subprocess,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--quick',action='store_true');p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 def run(cmd):return json.loads(subprocess.check_output(list(map(str,cmd)),text=True))
 worker=REPO/'target/release/fcmaes-native-worker';runner=REPO/'target/release/fcmaes-run';rows=[]
 env={'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,'build':run([worker,'describe'])['build'],'quick_smoke':a.quick}
 (a.output/'environment.json').write_text(json.dumps(env,indent=2))
 kinds=['sphere-bite','cassini1-bite'];counts=[1,2] if a.quick else [1,2,4,8,16];repeats=1 if a.quick else a.repeats
 for kind in kinds:
  for n in counts:
   for length in (['short'] if a.quick else ['short','long']):
    for repeat in range(repeats):
     s=json.loads((REPO/f'docs/runner-jobs/{kind}.json').read_text());s['execution'].update(workers=n,retry_wave_size=32)
     s['budget'].update(retries=8 if a.quick else (160 if length=='short' else 16),evaluations_per_retry=1000 if a.quick else (100000 if length=='short' else 1000000),wall_time_seconds=86400)
     label=f'{kind}-w{n}-{length}-r{repeat}';s['request_key']=label;spec=a.output/f'{label}.json';spec.write_text(json.dumps(s))
     modes=['none','shared_atomic','journal'];random.Random(repeat).shuffle(modes)
     for mode in modes:
      out=a.output/f'{label}-{mode}';r=run([worker,'benchmark',spec,out,mode]);rows.append({'problem':kind,'workers':n,'length':length,'repeat':repeat,'mode':mode,'actual_evaluations':r['actual_evaluations'],'wall_seconds':r['wall_seconds'],'evals_per_second':r['evaluations_per_second']})
 # Direct and managed full worker path includes snapshot publication; isolate node state.
 root=a.output/'node';run([runner,'--root',root,'node','init','--capacity','16']);run([runner,'--root',root,'workers','add','--id','native','--executable',worker])
 spec=a.output/'launch.json';s=json.loads((REPO/'docs/runner-jobs/sphere-bite.json').read_text());s['request_key']='launch-comparison';s['budget']['evaluations_per_retry']=1000 if a.quick else 1000000;spec.write_text(json.dumps(s))
 for mode in ['direct','managed']:
  t=time.perf_counter()
  if mode=='direct':
   out=a.output/'direct';out.mkdir();(out/'resolved.json').write_text(json.dumps(s));subprocess.run([str(worker),'execute',str(out.resolve())],check=True)
  else:out=Path(run([runner,'--root',root,'run','--spec',spec])['directory'])
  seconds=time.perf_counter()-t;r=json.loads((out/'run.json').read_text());rows.append({'problem':'sphere-bite','workers':2,'length':'launch','repeat':0,'mode':mode,'actual_evaluations':r['actual_evaluations'],'wall_seconds':seconds,'evals_per_second':r['actual_evaluations']/seconds})
 with (a.output/'timings.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'output':str(a.output),'target_gate':'Require >=30s measured jobs, paired repeats, 16 physical cores; quick output cannot pass the <=2% gate. Full incumbent CAS and expensive-simulator variants remain future adapters.'}))
if __name__=='__main__':main()
