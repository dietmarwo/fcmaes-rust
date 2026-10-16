#!/usr/bin/env python3
"""Real native binaries, isolated directories, no third-party Python dependencies."""
import copy, hashlib, json, os, signal, subprocess, tempfile, time, unittest
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
RUN=REPO/'target/release/fcmaes-run'
WORKER=REPO/'target/release/fcmaes-native-worker'
BASE=json.loads((REPO/'docs/runner-jobs/sphere-bite.json').read_text())

def records(directory):
    out=[]
    for line in (directory/'retries.jsonl').read_bytes().splitlines():
        try: e=json.loads(line)
        except json.JSONDecodeError: break
        assert e['checksum']==hashlib.sha256(e['payload_json'].encode()).hexdigest()
        out.append(json.loads(e['payload_json']))
    return sorted(out,key=lambda r:r['retry_id'])

class Integration(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'node';self.counter=0
        self.call('node','init','--capacity','16')
        self.call('workers','add','--id','native','--executable',WORKER)
    def call(self,*args,ok=True):
        p=subprocess.run([str(RUN),'--root',str(self.root),*map(str,args)],capture_output=True,text=True,timeout=30)
        if ok:self.assertEqual(p.returncode,0,p.stderr);return json.loads(p.stdout)
        self.assertNotEqual(p.returncode,0,p.stdout);return p
    def job(self,**changes):
        s=copy.deepcopy(BASE);self.counter+=1;s['request_key']=f'test-{self.counter}'
        for path,value in changes.items():
            at=s;keys=path.split('__')
            for k in keys[:-1]:at=at[k]
            at[keys[-1]]=value
        p=self.base/f'job-{self.counter}.json';p.write_text(json.dumps(s));return p,s
    def run_job(self,**changes):
        p,s=self.job(**changes);j=self.call('run','--spec',p);d=Path(j['directory']);return p,s,j,d,json.loads((d/'run.json').read_text())
    def start(self,**changes):
        p,s=self.job(**changes);proc=subprocess.Popen([str(RUN),'--root',str(self.root),'run','--spec',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.addCleanup(lambda:proc.poll() is None and proc.kill())
        id=hashlib.sha256(s['request_key'].encode()).hexdigest()[:32]
        d=self.root/'jobs'/id
        return proc,id,d
    def wait_state(self,d,wanted,seconds=10):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if (d/'job.json').exists():
                j=json.loads((d/'job.json').read_text())
                if j['state'] in wanted:return j
            time.sleep(.01)
        self.fail(f'job did not reach {wanted}: {d}')
    def test_telemetry_benchmark_counts(self):
        p,s=self.job(budget__retries=3,budget__evaluations_per_retry=123)
        for mode in ['none','shared_atomic','journal']:
            out=self.base/mode
            r=json.loads(subprocess.check_output([str(WORKER),'benchmark',str(p),str(out),mode],text=True))
            self.assertEqual(r['actual_evaluations'],369)
            self.assertEqual(r['shared_counter'],369 if mode=='shared_atomic' else 0)
            if mode=='journal':self.assertEqual(len(records(out)),3)
    def test_validation_and_capability_boundaries(self):
        for changes in [dict(execution__workers=0),dict(execution__retry_wave_size=0),dict(budget__retries=0),dict(budget__evaluations_per_retry=0),dict(optimizer__kind='de'),dict(problem__bounds__lower=[6]),dict(problem__config__dimension=0),dict(budget__retries=2**63,budget__evaluations_per_retry=4),dict(budget__wall_time_seconds=-1)]:
            p,_=self.job(**changes);self.call('validate','--spec',p,ok=False)
        p,s=self.job();s['bogus']=1;p.write_text(json.dumps(s));self.call('validate','--spec',p,ok=False)
        p,_=self.job(optimizer__kind='de',budget__evaluation_limit_mode='soft',optimizer__parameters={'population':7})
        r=self.call('validate','--spec',p);self.assertEqual(r['maximum_actual_evaluations'],r['requested_evaluations']+8*13)
    def test_bite_hard_and_de_soft_awkward_budgets(self):
        for algo in ['biteopt','de']:
            for budget in [1,2,6,7,8,17,33,101]:
                _,s,j,d,r=self.run_job(optimizer__kind=algo,optimizer__parameters={'population':7},budget__evaluation_limit_mode='hard' if algo=='biteopt' else 'soft',budget__evaluations_per_retry=budget,budget__retries=3)
                self.assertEqual(j['state'],'completed');rr=records(d);self.assertEqual(len(rr),3)
                for x in rr:self.assertLessEqual(x['actual_evaluations'],budget+(13 if algo=='de' else 0))
                self.assertEqual(r['actual_evaluations'],sum(x['actual_evaluations'] for x in rr))
    def test_retries_are_identical_across_workers_and_waves(self):
        for algo in ['biteopt','de']:
            results=[]
            for workers,wave in [(1,1),(2,5),(16,16)]:
                _,_,_,d,_=self.run_job(optimizer__kind=algo,budget__evaluation_limit_mode='hard' if algo=='biteopt' else 'soft',budget__retries=17,budget__evaluations_per_retry=401,execution__workers=workers,execution__retry_wave_size=wave)
                results.append([{k:r[k] for k in ['retry_id','run_seed','initialization','x','y','actual_evaluations']}for r in records(d)])
            self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])
    def test_idempotency_and_normalization(self):
        p,s,j,d,r=self.run_job();before=(d/'retries.jsonl').read_bytes()
        self.assertEqual(self.call('run','--spec',p)['id'],j['id']);self.assertEqual(before,(d/'retries.jsonl').read_bytes())
        s['problem']['bounds']={'lower':[-5.]*10,'upper':[5.]*10,'broadcast':False};p.write_text(json.dumps(s));self.call('run','--spec',p)
        s['execution']['seed']=43;p.write_text(json.dumps(s));self.call('run','--spec',p,ok=False)
    def test_gtop_native_and_comparison(self):
        p=REPO/'docs/runner-jobs/cassini1-bite.json';j=self.call('run','--spec',p);r=json.loads((Path(j['directory'])/'run.json').read_text());self.assertIsNotNone(r['best']);self.assertEqual(r['status'],'completed')
        _,_,_,a,ra=self.run_job();_,_,_,b,rb=self.run_job(optimizer__kind='de',budget__evaluation_limit_mode='soft')
        comp=self.call('compare',a,b);self.assertTrue(comp['equal_requested_work']);self.assertEqual(len(comp['arms']),2)
        self.call('compare',a,j['directory'],ok=False)
        # Legacy results remain readable with explicit unresolved-domain warning.
        for obj in [ra,rb]:obj.pop('runner');obj['tutorial']='legacy';obj['formulation']='mo';obj['best']=None;obj['feasible_results']=0
        fa=self.base/'legacy1.json';fb=self.base/'legacy2.json';fa.write_text(json.dumps(ra));fb.write_text(json.dumps(rb))
        c=self.call('compare',fa,fb);self.assertEqual(c['arms'][0]['feasible_results'],0);self.assertTrue(c['warnings'])
    def test_timeout_forced_accounting_is_unknown(self):
        # One second leaves the worker time to enter its endless wave before the deadline;
        # a worker still starting would stop cleanly at its first wave boundary instead.
        _,_,j,d,r=self.run_job(budget__evaluations_per_retry=10**9,budget__retries=2,budget__wall_time_seconds=1.0)
        self.assertEqual(j['state'],'timed_out');self.assertIsNone(r['actual_evaluations']);self.assertFalse(r['runner']['accounting_complete'])
        self.assertEqual(self.call('jobs','reconcile'),{})
    def test_cancel_preserves_completed_work(self):
        proc,id,d=self.start(budget__retries=100000,budget__evaluations_per_retry=1000,execution__retry_wave_size=4)
        self.wait_state(d,{'running'})
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            if (d/'retries.jsonl').exists() and len(records(d))>=8:break
            time.sleep(.01)
        self.call('jobs','cancel',id);out,err=proc.communicate(timeout=10);self.assertEqual(proc.returncode,0,err)
        r=json.loads((d/'run.json').read_text());self.assertEqual(r['status'],'cancelled');self.assertGreater(r['runner']['completed_retries'],0)
        self.assertEqual(self.call('jobs','reconcile'),{})
    def test_three_jobs_respect_capacity_and_waiting_cancel(self):
        # Independent root configured with two slots; no mutation of live capacity.
        self.root=self.base/'small';self.call('node','init','--capacity','2');self.call('workers','add','--id','native','--executable',WORKER)
        a,aid,ad=self.start(execution__workers=1,budget__evaluations_per_retry=10**9,budget__wall_time_seconds=20)
        self.wait_state(ad,{'running'})
        b,bid,bd=self.start(execution__workers=1,budget__evaluations_per_retry=10**9,budget__wall_time_seconds=20)
        self.wait_state(bd,{'running'})
        c,cid,cd=self.start(execution__workers=1,budget__evaluations_per_retry=10**9,budget__wall_time_seconds=20)
        self.wait_state(cd,{'waiting'});time.sleep(.15)
        ledger=json.loads((self.root/'slots.json').read_text());self.assertEqual(sum(x['slots'] for x in ledger.values()),2);self.assertFalse((cd/'retries.jsonl').exists())
        self.call('jobs','cancel',cid);c.communicate(timeout=10);self.assertEqual(json.loads((cd/'run.json').read_text())['actual_evaluations'],0)
        # Endless retries never finish their wave: a graceful cancel would wait, --now kills.
        for p,id in [(a,aid),(b,bid)]:self.call('jobs','cancel',id,'--now');p.communicate(timeout=10)
        self.assertEqual(self.call('jobs','reconcile'),{})
    def test_duplicate_journal_and_artifact_escape(self):
        _,_,j,d,_=self.run_job();p=d/'retries.jsonl';data=p.read_bytes();p.write_bytes(data+data.splitlines(keepends=True)[0]);self.call('jobs','get',j['id'],ok=False);p.write_bytes(data)
        run=json.loads((d/'run.json').read_text());run['artifacts']['bad']='../outside';(d/'run.json').write_text(json.dumps(run));(d.parent/'outside').write_text('private');self.call('jobs','artifact',j['id'],'bad',ok=False)
    def test_worker_kill_preserves_journal_and_marks_incomplete(self):
        proc,id,d=self.start(budget__retries=100000,budget__evaluations_per_retry=1000)
        self.wait_state(d,{'running'})
        end=time.monotonic()+10
        while time.monotonic()<end:
            if (d/'retries.jsonl').exists() and len(records(d))>=8:break
            time.sleep(.01)
        children=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                argv=(entry/'cmdline').read_bytes().split(b'\0')
                if argv[0]==str(d/'worker').encode():
                    st=(entry/'status').read_text()
                    children.append(int(next(l for l in st.splitlines() if l.startswith('NSpid:')).split()[-1]))
            except (OSError,StopIteration):pass
        self.assertEqual(len(children),1)
        os.kill(int(children[0]),signal.SIGKILL)
        out,err=proc.communicate(timeout=10);self.assertEqual(proc.returncode,0,err)
        r=json.loads((d/'run.json').read_text());self.assertEqual(r['status'],'failed');self.assertIsNone(r['actual_evaluations'])
        self.assertGreater(r['runner']['completed_retries'],0);self.assertEqual(self.call('jobs','reconcile'),{})
    def test_unknown_liveness_retains_slots_and_old_boot_releases(self):
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        record={'slots':16,'boot_id':boot,'launcher_pid':99999999,'unit':None}
        ledger=self.root/'slots.json';ledger.write_text(json.dumps({'unknown':record}))
        self.assertEqual(self.call('jobs','reconcile')['unknown'],record)
        record['boot_id']='previous-test-boot';ledger.write_text(json.dumps({'unknown':record}));self.assertEqual(self.call('jobs','reconcile'),{})
    def test_real_tutorial_manifests_compare(self):
        p=REPO/'tutorials/gtoc1-route-search/results/live-l0-seed42'
        c=self.call('compare',p/'agent/run.json',p/'evolutionary/run.json')
        self.assertEqual(len(c['arms']),2);self.assertTrue(c['warnings'])
    def test_unavailable_systemd_never_claims_submission(self):
        if self.call('node','doctor')['systemd_user_manager']:self.skipTest('real service host: use test_systemd.py')
        p,_=self.job();self.call('submit','--spec',p,ok=False);self.assertEqual(self.call('jobs','list'),[])

if __name__=='__main__':unittest.main(verbosity=2)
