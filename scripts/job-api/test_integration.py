#!/usr/bin/env python3
"""Black-box lifecycle and numerical acceptance tests. Python stdlib only.

Run from any repository after cargo build --release. Pass other workers to
exercise mixed-solver admission on a single node.
"""
import argparse, copy, hashlib, json, math, os, signal, subprocess, tempfile, time, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--runner', type=Path, default=REPO/'target/release/optimization-run')
for name, binary in [('fcmaes', 'fcmaes-job-worker'), ('lp_ip', 'lp-ip-job-worker'), ('sqp', 'sqp-job-worker')]:
    PARSER.add_argument('--'+name.replace('_','-'), type=Path, default=REPO/'target/release'/binary)
ARGS, REST = PARSER.parse_known_args()
WORKERS = {name:p.resolve() for name,p in [('fcmaes',ARGS.fcmaes),('lp-ip',ARGS.lp_ip),('sqp',ARGS.sqp)] if p.is_file()}
RUN = ARGS.runner.resolve()


def envelope(worker, payload):
    return {'schema_version':'optimization.job/0.1','request_key':'test', 'worker':worker,
            'payload_schema':{'fcmaes':'fcmaes.retry/1','lp-ip':'lp-ip.solve/1','sqp':'sqp.solve/1'}.get(worker,'test.sleep/1'),
            'payload':payload,'resources':{'cpu_slots':8,'outer_workers':8,'inner_threads':1},
            'limits':{'wall_time_seconds':30.,'grace_seconds':.2},'inputs':[]}


class Jobs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.root=self.base/'node'; self.counter=0; self.processes=[]
        self.addCleanup(self.stop_processes)
        self.call('node','init','--capacity',8)
        for name,exe in WORKERS.items(): self.call('workers','add','--id',name,'--executable',exe)
    def stop_processes(self):
        for proc,d in self.processes:
            path=d/'worker-process.json'
            if path.exists():
                try: os.killpg(json.loads(path.read_text())['pgid'], signal.SIGKILL)
                except ProcessLookupError: pass
            if proc.poll() is None: proc.kill()
            proc.communicate(timeout=5)
    def call(self,*args,ok=True):
        p=subprocess.run([str(RUN),'--root',str(self.root),*map(str,args)],capture_output=True,text=True,timeout=45)
        if ok:
            self.assertEqual(p.returncode,0,p.stderr);return json.loads(p.stdout)
        self.assertNotEqual(p.returncode,0,p.stdout);return p
    def write(self,s):
        self.counter+=1;s=copy.deepcopy(s);s['request_key']=f'job-{self.counter}'
        path=self.base/f'spec-{self.counter}.json';path.write_text(json.dumps(s));return path,s
    def fixture(self,name):
        if name=='fcmaes':
            return envelope(name,{'problem':{'id':'benchmark.sphere','version':'1','config':{'dimension':4},'bounds':{'lower':[-5.],'upper':[5.],'broadcast':True}},'optimizer':{'kind':'biteopt','parameters':{'population':12}},'seed':42,'retries':16,'evaluations_per_retry':1000,'evaluation_limit_mode':'hard'})
        if name=='sqp':
            return envelope(name,{'problem':'benchmark.simplex','dimension':4,'starts':[[.1,.2,.3,.4]]*16,'backend':'active_set','max_iterations':100,'max_callback_calls':2000,'feasibility_tolerance':1e-7,'stationarity_tolerance':1e-6})
        s=envelope(name,{'model_input':'model','starts':[None]*16,'node_limit':10000,'iteration_limit':100000,'feasibility_tolerance':1e-7,'integrality_tolerance':1e-7,'presolve':True})
        self.model(s,{'schema_version':'lp-ip.model/1','sense':'minimize','objective':[2.,3.],'objective_offset':5.,'variable_lower':[0.,0.],'variable_upper':[1.,1.],'kinds':['binary','binary'],'matrix':{'indptr':[0,2],'indices':[0,1],'values':[1.,1.]},'row_lower':[1.],'row_upper':[1.]})
        return s
    def model(self,s,data):
        blob=json.dumps(data,sort_keys=True).encode();name='model-'+hashlib.sha256(blob).hexdigest()+'.json';(self.base/name).write_bytes(blob)
        s['inputs']=[{'name':'model','file':name,'sha256':hashlib.sha256(blob).hexdigest()}]
    def start(self,s):
        path,s=self.write(s);ident=hashlib.sha256(s['request_key'].encode()).hexdigest()[:32];d=self.root/'jobs'/ident
        p=subprocess.Popen([str(RUN),'--root',str(self.root),'run','--spec',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.processes.append((p,d));return p,ident,d
    def state(self,d,wanted,timeout=12):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if (d/'job.json').exists():
                value=json.loads((d/'job.json').read_text())
                if value['state'] in wanted:return value
            time.sleep(.01)
        self.fail(f'No state {wanted} for {d}')
    def run_spec(self,s):
        path,s=self.write(s);job=self.call('run','--spec',path);d=Path(job['directory']);run=json.loads((d/'run.json').read_text());return path,s,job,d,run
    def fake(self):
        p=self.base/'fake-worker'
        p.write_text('''#!/usr/bin/python3
import sys,json,time,fcntl
from pathlib import Path
if sys.argv[1]=='describe': print(json.dumps({'protocol':'optimization.worker/0.1','profiles':['test.sleep/1']}))
elif sys.argv[1]=='validate':
 s=json.loads(Path(sys.argv[2]).read_text());s['task_count']=1;print(json.dumps({'resolved':s}))
else:
 d=Path(sys.argv[2]);f=(d/'worker.lock').open('w');fcntl.flock(f,fcntl.LOCK_EX)
 if json.loads((d/'resolved.json').read_text())['payload'].get('false_complete'):
  s=json.loads((d/'resolved.json').read_text());s.pop('request_key');digest=__import__('hashlib').sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  (d/'run.json').write_text(json.dumps({'status':'completed','elapsed_seconds':0.,'results':[],'work':{},'runner':{'spec_digest':digest,'accounting_complete':True}}))
 else: time.sleep(120)
''');p.chmod(0o700);self.call('workers','add','--id','sleep','--executable',p)
        return envelope('sleep',{})
    def test_discovery_and_strict_validation(self):
        self.assertTrue(WORKERS,'build at least one real worker')
        for name in WORKERS:
            self.assertEqual(self.call('problems','describe',name)['protocol'],'optimization.worker/0.1')
            s=self.fixture(name);p,_=self.write(s);self.assertTrue(self.call('validate','--spec',p)['valid'])
            for mutate in [lambda s:s.update(extra=1),lambda s:s['resources'].update(inner_threads=8),lambda s:s.update(payload_schema='unknown/9'),lambda s:s['payload'].update(ignored_option=True)]:
                bad=copy.deepcopy(s);mutate(bad);p,_=self.write(bad);self.call('validate','--spec',p,ok=False)
    def test_actual_workers_numerics_journals_and_idempotency(self):
        for name in WORKERS:
            p,s,j,d,r=self.run_spec(self.fixture(name))
            self.assertEqual(j['state'],'completed');self.assertTrue(r['runner']['accounting_complete']);self.assertEqual(r['completed_tasks'],16)
            for rec in r['results']:
                o=rec['outcome'];self.assertTrue(o['feasible']);self.assertIsNotNone(o['objective'])
                if name=='lp-ip':self.assertAlmostEqual(o['objective'],7.);self.assertEqual(o['claim'],'numerical_optimum');self.assertAlmostEqual(o['details']['best_bound'],7.,places=10);self.assertLessEqual(o['details']['best_bound'],7.)
                if name=='sqp':self.assertAlmostEqual(o['objective'],.25,places=6);self.assertEqual(o['claim'],'first_order');self.assertLessEqual(o['details']['kkt']['stationarity'],1e-6)
                if name=='fcmaes':self.assertEqual(rec['work']['evaluations'],1000);self.assertEqual(o['claim'],'best_found');self.assertAlmostEqual(o['objective'],sum(x*x for x in o['x']))
            before=(d/'results.jsonl').read_bytes();self.assertEqual(self.call('run','--spec',p)['id'],j['id']);self.assertEqual(before,(d/'results.jsonl').read_bytes())
            self.assertEqual([e['state'] for e in self.call('jobs','events',j['id'])],['submitted','waiting','running','completed'])
            self.assertEqual(self.call('compare',d,d)['arms'][0]['status'],'completed')
            s['limits']['wall_time_seconds']=31.;p.write_text(json.dumps(s));self.call('run','--spec',p,ok=False)
            self.assertEqual(self.call('jobs','reconcile'),{})
    def test_one_and_eight_slots_preserve_numerical_results(self):
        for name in WORKERS:
            a=self.fixture(name);a['resources']={'cpu_slots':1,'outer_workers':1,'inner_threads':1}
            _,_,_,_,serial=self.run_spec(a)
            a['resources']={'cpu_slots':8,'outer_workers':8,'inner_threads':1}
            _,_,_,_,parallel=self.run_spec(a)
            for x,y in zip(serial['results'],parallel['results']):
                self.assertAlmostEqual(x['outcome']['objective'],y['outcome']['objective'],places=10)
                self.assertEqual(x['outcome']['termination'],y['outcome']['termination'])
                if name=='fcmaes':self.assertEqual(x['work'],y['work']);self.assertEqual(x['outcome']['x'],y['outcome']['x'])
            if name!='fcmaes':
                a['resources']={'cpu_slots':8,'outer_workers':1,'inner_threads':8}
                _,_,_,_,inner=self.run_spec(a)
                for x,y in zip(serial['results'],inner['results']):self.assertAlmostEqual(x['outcome']['objective'],y['outcome']['objective'],places=8)
    def test_deadline_on_each_real_worker(self):
        for name in WORKERS:
            s=self.fixture(name);s['limits']={'wall_time_seconds':.001,'grace_seconds':.1}
            if name=='fcmaes':s['payload']['evaluations_per_retry']=10**9
            else:s['payload']['starts']=s['payload']['starts']*100
            _,_,j,_,r=self.run_spec(s);self.assertEqual(j['state'],'timed_out');self.assertEqual(r['status'],'timed_out')
            self.assertLess(r['completed_tasks'],r['requested_tasks']);self.assertEqual(self.call('jobs','reconcile'),{})
    def test_shared_capacity_waiting_cancel_and_force_timeout(self):
        s=self.fake();s['limits']={'wall_time_seconds':20.,'grace_seconds':.05}
        p,ident,d=self.start(s);self.state(d,{'running'})
        for name in WORKERS:
            q,qid,qd=self.start(self.fixture(name));self.state(qd,{'waiting'});self.assertFalse((qd/'results.jsonl').exists())
            slots=json.loads((self.root/'slots.json').read_text());self.assertEqual(sum(v['slots'] for v in slots.values()),8)
            self.call('jobs','cancel',qid);out,err=q.communicate(timeout=10);self.assertEqual(q.returncode,0,err)
            result=json.loads((qd/'run.json').read_text());self.assertEqual(result['status'],'cancelled');self.assertEqual(result['work'],{})
        self.call('jobs','cancel',ident,'--now');p.communicate(timeout=10);self.assertEqual(json.loads((d/'run.json').read_text())['status'],'cancelled')
        s['limits']={'wall_time_seconds':.1,'grace_seconds':.05};_,_,j,_,r=self.run_spec(s)
        self.assertEqual(j['state'],'timed_out');self.assertFalse(r['runner']['accounting_complete']);self.assertIsNone(r['work']);self.assertEqual(self.call('jobs','reconcile'),{})
    def test_launcher_loss_retains_live_worker_then_repairs_without_resume(self):
        s=self.fake();p,ident,d=self.start(s);self.state(d,{'running'})
        end=time.monotonic()+5
        while not (d/'worker-process.json').exists() and time.monotonic()<end:time.sleep(.01)
        p.kill();p.communicate(timeout=5)
        self.assertIn(ident,self.call('jobs','reconcile'));self.call('jobs','repair',ident,ok=False)
        os.killpg(json.loads((d/'worker-process.json').read_text())['pgid'],signal.SIGKILL)
        end=time.monotonic()+5
        while time.monotonic()<end:
            repaired=subprocess.run([str(RUN),'--root',str(self.root),'jobs','repair',ident],capture_output=True,text=True)
            if repaired.returncode==0:break
            time.sleep(.02)
        self.assertEqual(repaired.returncode,0,repaired.stderr);r=json.loads((d/'run.json').read_text());self.assertEqual(r['status'],'interrupted');self.assertIsNone(r['work']);self.assertIsNone(r['elapsed_seconds']);self.assertEqual(self.call('jobs','reconcile'),{})
    def test_launcher_lost_while_queued_is_sealed_with_zero_work(self):
        # A holder fills the node; the queued job's launcher dies before admission.
        hold=self.fake();_,_,hold_dir=self.start(hold);self.state(hold_dir,{'running'})
        p,ident,d=self.start(self.fake());self.state(d,{'waiting'})
        p.kill();p.communicate(timeout=5)
        self.assertEqual(self.call('jobs','get',ident)['observed_state'],'launcher_lost')
        # No worker ever ran, so repair needs no liveness proof and no reservation exists.
        self.assertEqual(self.call('jobs','repair',ident)['state'],'interrupted')
        r=json.loads((d/'run.json').read_text());self.assertEqual(r['work'],{});self.assertTrue(r['runner']['accounting_complete'])
        self.assertFalse((d/'worker-process.json').exists());self.assertNotIn(ident,self.call('jobs','reconcile'))
        self.assertEqual(self.call('jobs','get',ident)['state'],'interrupted')
    def test_artifact_corruption_and_escape(self):
        name=next(iter(WORKERS));_,_,j,d,r=self.run_spec(self.fixture(name));path=d/'results.jsonl';data=path.read_bytes();path.write_bytes(data+data.splitlines(keepends=True)[0]);self.call('jobs','get',j['id'],ok=False);path.write_bytes(data)
        r['artifacts']['bad']='../outside';(d.parent/'outside').write_text('outside');(d/'run.json').write_text(json.dumps(r));self.call('jobs','artifact',j['id'],'bad',ok=False)
        if 'lp-ip' in WORKERS:
            s=self.fixture('lp-ip');model=self.base/s['inputs'][0]['file'];model.write_text('{}');p,_=self.write(s);self.call('validate','--spec',p,ok=False)
    def test_lp_lp_mip_infeasible_unbounded_and_validation(self):
        if 'lp-ip' not in WORKERS:self.skipTest('LP worker not supplied')
        s=self.fixture('lp-ip');s['payload']['starts']=[None]
        model=json.loads((self.base/s['inputs'][0]['file']).read_text());model['kinds']=['continuous']*2;self.model(s,model)
        _,_,_,_,r=self.run_spec(s);self.assertEqual(r['results'][0]['outcome']['termination'],'optimal');self.assertIsNotNone(r['results'][0]['outcome']['details']['lp_residuals'])
        model['row_lower']=[3.];model['row_upper']=[3.];self.model(s,model)
        _,_,j,_,r=self.run_spec(s);self.assertEqual(j['state'],'completed');self.assertEqual(r['results'][0]['outcome']['termination'],'infeasible');self.assertEqual(r['results'][0]['outcome']['claim'],'none')
        model.update(objective=[-1.,0.],variable_upper=[None,None],row_lower=[],row_upper=[],matrix={'indptr':[0],'indices':[],'values':[]});self.model(s,model)
        _,_,_,_,r=self.run_spec(s);self.assertEqual(r['results'][0]['outcome']['termination'],'unbounded')
        model['matrix']['indptr']=[1];self.model(s,model);p,_=self.write(s);self.call('validate','--spec',p,ok=False)
    def test_sqp_backends_and_initial_point_validation(self):
        if 'sqp' not in WORKERS:self.skipTest('SQP worker not supplied')
        for backend in ['active_set','clarabel']:
            for problem,start,expected in [('benchmark.quadratic',[-2.,3.],0.),('benchmark.simplex',[0.,0.],.5),('benchmark.rosenbrock',[-1.2,1.],0.)]:
                s=self.fixture('sqp');s['payload'].update(problem=problem,dimension=2,starts=[start],backend=backend,max_iterations=200)
                _,_,_,_,r=self.run_spec(s);o=r['results'][0]['outcome'];self.assertEqual(o['claim'],'first_order',o);self.assertAlmostEqual(o['objective'],expected,places=5)
        s['payload']['starts']=[[99.,0.]];p,_=self.write(s);self.call('validate','--spec',p,ok=False)
    def test_fcmaes_soft_budget_and_inner_threads_rejection(self):
        if 'fcmaes' not in WORKERS:self.skipTest('fcmaes worker not supplied')
        s=self.fixture('fcmaes');s['payload']['optimizer']['kind']='de';s['payload']['evaluation_limit_mode']='soft';s['payload']['evaluations_per_retry']=7
        _,_,_,_,r=self.run_spec(s)
        for rec in r['results']:self.assertLessEqual(rec['work']['evaluations'],7+2*12-1)
        s['resources']={'cpu_slots':8,'outer_workers':1,'inner_threads':8};p,_=self.write(s);self.call('validate','--spec',p,ok=False)

    def test_fcmaes_cmaes_mode_and_map_elites_profiles(self):
        if 'fcmaes' not in WORKERS:self.skipTest('fcmaes worker not supplied')
        s=self.fixture('fcmaes');s['payload']['optimizer']={'kind':'cmaes','parameters':{'population':8}};s['payload']['evaluation_limit_mode']='soft'
        _,_,j,_,r=self.run_spec(s);self.assertEqual(j['state'],'completed')
        for rec in r['results']:
            o=rec['outcome'];self.assertEqual(o['claim'],'best_found');self.assertLessEqual(rec['work']['evaluations'],1000+7);self.assertAlmostEqual(o['objective'],sum(x*x for x in o['x']))
        mo=envelope('fcmaes',{'problem':{'id':'benchmark.zdt1','version':'1','config':{'dimension':4},'bounds':{'lower':[0.],'upper':[1.],'broadcast':True}},'optimizer':{'kind':'mode','parameters':{'population':16}},'seed':3,'runs':4,'generations':40})
        mo['payload_schema']='fcmaes.mode/1'
        _,_,j,_,r=self.run_spec(mo);self.assertEqual(j['state'],'completed');self.assertEqual(r['work']['evaluations'],4*40*16)
        for rec in r['results']:
            o=rec['outcome'];self.assertEqual(o['claim'],'non_dominated_set');self.assertIsNone(o['x']);front=o['details']['front'];self.assertTrue(front)
            for point in front:
                x=point['x'];g=1+9*sum(x[1:])/(len(x)-1)
                for got,want in zip(point['objectives'],[x[0],g*(1-math.sqrt(x[0]/g))]):self.assertAlmostEqual(got,want,places=12)
            for a in front:
                self.assertFalse(any(all(u<=v for u,v in zip(b['objectives'],a['objectives'])) and b['objectives']!=a['objectives'] for b in front))
            self.assertGreater(o['details']['hypervolume']['value'],0.)
        qd=envelope('fcmaes',{'problem':{'id':'benchmark.sphere-projection','version':'1','config':{'dimension':4},'bounds':{'lower':[-5.12],'upper':[5.12],'broadcast':True}},'archive':{'capacity':25},'optimizer':{'kind':'map_elites','parameters':{'chunk_size':8}},'seed':5,'runs':4,'generations':30})
        qd['payload_schema']='fcmaes.map-elites/1'
        _,_,j,_,r=self.run_spec(qd);self.assertEqual(j['state'],'completed');self.assertEqual(r['work']['evaluations'],4*30*8)
        for rec in r['results']:
            o=rec['outcome'];self.assertEqual(o['claim'],'elite_archive');elites=o['details']['elites'];self.assertEqual(len(elites),o['details']['occupied'])
            for e in elites:
                x=e['x'];self.assertAlmostEqual(e['fitness'],sum((v-.4*5.12)**2 for v in x),places=10)
                self.assertAlmostEqual(e['descriptor'][0],sum(x[:2]),places=12);self.assertAlmostEqual(e['descriptor'][1],sum(x[2:]),places=12)
            self.assertEqual(o['objective'],min(e['fitness'] for e in elites))
        mo['payload']['generations']=10**9;mo['limits']={'wall_time_seconds':.3,'grace_seconds':.5}
        _,_,j,_,r=self.run_spec(mo);self.assertEqual(j['state'],'timed_out');self.assertTrue(r['runner']['accounting_complete'])
        for rec in r['results']:self.assertEqual(rec['outcome']['termination'],'cancelled');self.assertEqual(rec['outcome']['claim'],'non_dominated_set')

if __name__=='__main__':unittest.main(argv=[__file__,*REST],verbosity=2)
