#!/usr/bin/env python3
"""Real runner/MCP smoke and cooperative simulation cancellation; stdlib only."""
import argparse, copy, hashlib, json, os, queue, subprocess, tempfile, threading, time, unittest
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
PARSER=argparse.ArgumentParser()
PARSER.add_argument('--bin-dir',type=Path,default=Path(os.environ.get('CARGO_TARGET_DIR',REPO/'target'))/'release')
ARGS,REST=PARSER.parse_known_args();BINS=ARGS.bin_dir.resolve()

class Client:
    def __init__(self,root):
        self.process=subprocess.Popen([str(BINS/'optimization-mcp'),'--root',str(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        self.messages=queue.Queue();self.ident=0
        def read():
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
        self.request('initialize',{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'surrogate-test','version':'1'}})
    def request(self,method,params):
        self.ident+=1;ident=self.ident
        self.process.stdin.write(json.dumps({'jsonrpc':'2.0','id':ident,'method':method,'params':params})+'\n');self.process.stdin.flush()
        while True:
            reply=self.messages.get(timeout=60)
            if reply.get('id')==ident:
                if 'error' in reply:raise AssertionError(reply)
                return reply['result']
    def tool(self,name,arguments):
        reply=self.request('tools/call',{'name':name,'arguments':arguments})
        if reply.get('isError'):raise AssertionError(reply)
        return reply['structuredContent']
    def close(self):
        self.process.stdin.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        self.process.stdout.close();self.process.stderr.close()

class Jobs(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'node'
        self.call('node','init','--capacity',8)
        self.call('workers','add','--id','fcmaes','--executable',BINS/'fcmaes-job-worker')
        self.call('workers','add','--id','simulation','--executable',BINS/'examples/simulation_worker')
    def call(self,*args):
        p=subprocess.run([str(BINS/'optimization-run'),'--root',str(self.root),*map(str,args)],capture_output=True,text=True,timeout=60)
        self.assertEqual(p.returncode,0,p.stderr);return json.loads(p.stdout)
    def test_mcp_submits_a_real_surrogate_job_and_reports_evidence(self):
        spec=json.loads((REPO/'tutorials/surrogate/branin.json').read_text());spec['payload']['evaluations']=12
        spec['payload']['parameters']={'initial_points':4,'candidates':64,'fit_restarts':1,'fit_iterations':10}
        client=Client(self.root);self.addCleanup(client.close)
        description=client.tool('describe_worker',{'worker':'fcmaes'});self.assertIn('fcmaes.surrogate/1',description['profiles'])
        ident=client.tool('submit_job',{'job':spec})['job_id']
        end=client.tool('wait_job',{'job_id':ident,'timeout_seconds':30});self.assertEqual(end['state'],'completed')
        result=client.tool('get_result',{'job_id':ident});self.assertTrue(result['accounting_complete'])
        self.assertEqual(result['tasks'][0]['claim'],'best_found');self.assertEqual(result['tasks'][0]['evidence'],'rechecked')
        manifest=self.call('result',self.root/'jobs'/ident);self.assertGreater(manifest['work']['model_fits'],0)
        # The trace and the snapshot are declared artifacts: indexed with their
        # SHA-256 and readable through MCP, the trace in pages.
        snapshot=result['artifact_index']['surrogate-observations-0']
        data=(self.root/'jobs'/ident/snapshot['file']).read_bytes();self.assertEqual(snapshot['sha256'],hashlib.sha256(data).hexdigest())
        records,offset=[],0
        while True:
            page=client.tool('get_artifact',{'job_id':ident,'name':'surrogate-0','offset':offset,'limit':10});records+=page['records'];offset=page['next_offset']
            if offset>=page['available']:break
        self.assertEqual(sum(r['record']=='event' and r['data']['event']=='completed' for r in records),12)
        self.assertEqual(json.loads(client.tool('get_artifact',{'job_id':ident,'name':'surrogate-observations-0'})['content'])['observations'][0]['id'],0)
    def test_runner_cancels_in_flight_rust_simulations_without_failing_them(self):
        spec=json.loads((REPO/'tutorials/surrogate/oscillator.json').read_text());spec['request_key']='surrogate-cancel'
        spec['payload']['problem']['parameters'].update(steps=100000,pause_ms=1)
        spec['payload']['concurrent']=2;spec['resources'].update(cpu_slots=4,inner_threads=4)
        spec['limits']={'wall_time_seconds':30.0,'grace_seconds':2.0}
        path=self.base/'cancel.json';path.write_text(json.dumps(spec));ident=hashlib.sha256(spec['request_key'].encode()).hexdigest()[:32]
        directory=self.root/'jobs'/ident
        process=subprocess.Popen([str(BINS/'optimization-run'),'--root',str(self.root),'run','--spec',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+15;started=False
            # The running job's trace, read as a declared artifact.
            while time.monotonic()<deadline and not started:
                page=subprocess.run([str(BINS/'optimization-run'),'--root',str(self.root),'jobs','artifact',ident,'surrogate-0','--offset','0','--limit','10000'],capture_output=True,text=True)
                if page.returncode==0:
                    page=json.loads(page.stdout);self.assertFalse(page['final'])
                    started=any(r['record']=='event' and r['data']['event']=='started' for r in page['records'])
                time.sleep(.005)
            self.assertTrue(started,'simulation did not start')
            before=time.monotonic();self.call('jobs','cancel',ident)
            out,err=process.communicate(timeout=6);self.assertEqual(process.returncode,0,err)
            self.assertLess(time.monotonic()-before,5)
            self.assertEqual(json.loads(out)['state'],'cancelled')
            run=self.call('result',directory);self.assertTrue(run['runner']['accounting_complete'])
            self.assertEqual(run['work']['failed_evaluations'],0)
            self.assertGreater(run['work']['cancelled_evaluations'],0)
        finally:
            if process.poll() is None:process.kill();process.communicate()

if __name__=='__main__':unittest.main(argv=[__file__,*REST],verbosity=2)
