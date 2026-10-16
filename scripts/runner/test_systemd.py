#!/usr/bin/env python3
"""Opt-in host gate: real systemd user services, never a mocked manager."""
import hashlib,json,subprocess,tempfile,time
from pathlib import Path
from test_integration import REPO,RUN,WORKER,BASE

def main():
 with tempfile.TemporaryDirectory(prefix='fcmaes-systemd-') as tmp:
  root=Path(tmp)/'node'
  def cli(*a):
   p=subprocess.run([str(RUN),'--root',str(root),*map(str,a)],text=True,capture_output=True,timeout=30)
   if p.returncode:raise RuntimeError(p.stderr)
   return json.loads(p.stdout)
  cli('node','init','--capacity','2')
  if not cli('node','doctor')['systemd_user_manager']:
   print('SKIP: no systemd user manager; detached acceptance NOT verified');return 77
  cli('workers','add','--id','native','--executable',WORKER)
  ids=[];pids=[]
  def wait_state(id,state):
   end=time.monotonic()+20
   while time.monotonic()<end:
    j=cli('jobs','get',id)
    if j['state']==state:return j
    time.sleep(.1)
   raise AssertionError(j)
  try:
   for i in range(3):
    s=json.loads(json.dumps(BASE));s['request_key']=f'systemd-host-{i}';s['execution']['workers']=1;s['budget']['evaluations_per_retry']=10**9;s['budget']['wall_time_seconds']=60
    spec=Path(tmp)/f'{i}.json';spec.write_text(json.dumps(s))
    # Each submitting client has exited on return: services must survive it.
    j=cli('submit','--spec',spec);ids.append(j['id'])
    assert cli('submit','--spec',spec)['id']==j['id']
    wait_state(j['id'],'running' if i<2 else 'waiting')
   slots=json.loads((root/'slots.json').read_text());assert sum(x['slots'] for x in slots.values())==2
   cli('jobs','cancel',ids[2]);wait_state(ids[2],'cancelled')
   cli('jobs','cancel',ids[0]);wait_state(ids[0],'cancelled')
   # Killing a launcher lets the systemd control group contain its worker.
   unit=f'fcmaes-{ids[1]}.service'
   subprocess.run(['systemctl','--user','kill','--kill-whom=main','--signal=KILL',unit],check=True)
   end=time.monotonic()+15
   while time.monotonic()<end:
    if ids[1] not in cli('jobs','reconcile'):break
    time.sleep(.2)
   else:raise AssertionError('crashed service reservation was not safely reconciled')
   j=cli('jobs','repair',ids[1]);assert j['state']=='interrupted'
   assert not cli('jobs','reconcile')
   print('PASS: three user services, idempotency, client exit, admission, waiting/active cancel, launcher kill, cgroup reconciliation and repair')
   print('OS-user logout/lingering and host reboot still require the manual protocol in docs/experiment-runner.md')
   return 0
  finally:
   for id in ids:subprocess.run(['systemctl','--user','stop',f'fcmaes-{id}.service'],capture_output=True,timeout=15)
if __name__=='__main__':raise SystemExit(main())
