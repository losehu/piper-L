"""Human takeover, live replay and atomic actor publication without hardware."""
import json,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from rlt_piper.hil import Handover
from rlt_piper.recording import Recorder,transitions
from rlt_piper.checkpoint import publish,snapshot,current
from rlt_piper.live import PriorityGate
from rlt_piper.contract import SCHEMA
from rlt_piper.serve import Service

CFG=dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05)
class HilTests(unittest.TestCase):
    def test_follower_freezes_before_leader_alignment_and_stale_results_rejected(self):
        class Align:
            error=None;result=None
            def start(self,target):self.target=np.array(target,copy=True)
        align=Align();h=Handover(align);q=np.arange(7)*.1
        leader=SimpleNamespace(vector=lambda:(q.copy(),np.ones(4)*10))
        previous=np.ones(7)*.02;h.toggle();self.assertFalse(h.accepts(0,11))
        with h.lock:h.switch(previous,leader,10)
        self.assertEqual(h.mode,'aligning');np.testing.assert_allclose(h.follow,previous)
        self.assertFalse(hasattr(align,'target'))
        for now in np.arange(10.01,10.8,.01):h.poll_alignment(previous,now)
        np.testing.assert_allclose(align.target,previous)
        q=previous.copy();q[6]=.08;align.result=(q.copy(),np.ones(4)*11)
        self.assertTrue(h.poll_alignment(previous,11));self.assertEqual(h.mode,'human')
        np.testing.assert_allclose(h.target(leader)[0],previous)
        q[0]+=.1;q[6]+=.005
        self.assertAlmostEqual(h.target(leader)[0][0],previous[0]+.1)
        self.assertAlmostEqual(h.target(leader)[0][6],previous[6]+.005)
        self.assertFalse(h.accepts(1,11.1));self.assertTrue(h.accepts(2,11.1))
        h.toggle();h.switch(previous,leader,12)
        self.assertEqual(h.mode,'policy');self.assertFalse(h.accepts(2,12.1));self.assertTrue(h.accepts(3,12.1))
    def test_leader_does_not_start_until_follower_stops(self):
        align=SimpleNamespace(error=None,result=None,start=lambda _:self.fail('Moving follower must not start leader'))
        h=Handover(align);h.switch(np.zeros(7),None,10)
        for i in range(200):
            now=10+i*.01;state=np.zeros(7);state[0]=np.radians(.3)*np.sin(i*.5)
            h.poll_alignment(state,now)
        self.assertFalse(h.align_started)
    def test_residual_servo_motion_settles_before_strict_hold_check(self):
        targets=[];align=SimpleNamespace(error=None,result=None,start=lambda q:targets.append(q.copy()))
        h=Handover(align);held=np.zeros(7);old=held.copy();old[1]=np.radians(3.035)
        h.switch(held,None,10,previous_command=old)
        # The observed failing run crossed 1 degree after just 60ms.
        for dt,offset in [(0.,0.),(.01,.156),(.02,.315),(.03,.281),(.04,.531),(.05,.736),(.06,.861),(.07,1.032)]:
            q=held.copy();q[1]=np.radians(offset);self.assertFalse(h.poll_alignment(q,10+dt))
        self.assertFalse(targets)
        # A stationary offset alone is insufficient: it must reach the hold target.
        for dt in np.arange(.08,.5,.01):
            q=held.copy();q[1]=np.radians(1.032);h.poll_alignment(q,10+dt)
        self.assertFalse(targets)
        for dt in np.arange(.5,1.2,.01):h.poll_alignment(held,10+dt)
        self.assertEqual(len(targets),1);np.testing.assert_array_equal(targets[0],held)
        q=held.copy();q[1]=np.radians(1.01)
        with self.assertRaisesRegex(RuntimeError,'保持姿态偏移'):h.poll_alignment(q,11.3)
    def test_residual_motion_outside_last_command_envelope_stops(self):
        align=SimpleNamespace(error=None,result=None,start=lambda _:self.fail('Leader started'))
        h=Handover(align);held=np.zeros(7);old=held.copy();old[1]=np.radians(3)
        h.switch(held,None,10,previous_command=old)
        q=held.copy();q[1]=np.radians(4.01)
        with self.assertRaisesRegex(RuntimeError,'停稳过程超出'):h.poll_alignment(q,10.06)
    def test_unsettled_follower_times_out_without_starting_leader(self):
        align=SimpleNamespace(error=None,result=None,start=lambda _:self.fail('Leader started'))
        h=Handover(align);held=np.zeros(7);old=held.copy();old[1]=np.radians(3)
        h.switch(held,None,10,previous_command=old)
        q=held.copy();q[1]=np.radians(1.1)
        with self.assertRaisesRegex(RuntimeError,'未稳定停止'):h.poll_alignment(q,13.01)
    def recorder(self,root,source='human',boundary=False):
        r=Recorder(root,'online','token',CFG)
        for i in range(16):
            t=10+i*.1
            result=dict(generation=1,rlt=dict(z_rl=np.ones(512)*i,proprio=np.zeros(7),ref_chunk=np.ones((18,7))*.02,
                        proposed_chunk=np.ones((18,7))*.9,source='base',actor_version=0))
            r.plan(SimpleNamespace(t=t),result,t+.14,source=source)
        for t in np.arange(9.99,11.601,1/120):
            gen=2 if boundary and t>10.65 else 1
            r.command(t,np.ones(7)*.01,source,gen)
        return r
    def test_live_human_is_bc_data_without_success_reward(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root);p=r.snapshot(live=True);rows=transitions(p,'token')
            self.assertTrue(rows)
            for row in rows:
                self.assertTrue(np.all(row['source_chunk']==2));self.assertFalse(row['done']);self.assertEqual(row['rewards'].sum(),0)
                np.testing.assert_allclose(row['action_chunk'],.01);np.testing.assert_allclose(row['ref_chunk'],.02)
            self.assertIsNone(json.loads((p/'episode.json').read_text())['outcome'])
    def test_terminal_replaces_live_reward_for_same_transition_identity(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root);p=r.snapshot(live=True);live={v['transition_id']:v for v in transitions(p,'token')}
            r.label=1;r.label_at=11.6
            terminal=dict(t=11.8,z_rl=np.zeros(512),proprio=np.zeros(7),ref_chunk=np.zeros((18,7)))
            r.save(terminal);final=transitions(p,'token')
            self.assertTrue(any(v['done'] and v['rewards'].sum()==1 for v in final))
            self.assertTrue(set(live)&{v['transition_id'] for v in final})
    def test_hold_and_cross_handover_windows_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root,source='hold');self.assertEqual(transitions(r.snapshot(live=True),'token'),[])
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root,boundary=True);rows=transitions(r.snapshot(live=True),'token')
            self.assertTrue(rows);self.assertTrue(all(float(v['transition_id'].split(':')[-1])<10.1 for v in rows))
    def test_aborted_human_retained_without_fabricated_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            r=self.recorder(root);rows=transitions(r.save(error='operator abort'),'token')
            self.assertTrue(rows);self.assertTrue(all(not v['done'] and not v['rewards'].any() for v in rows))
    def test_atomic_checkpoint_and_unready_not_executed(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);info=dict(token_sha256='token',actor_updates=2,global_step=4,ready=False)
            payload=dict(version=2,actor_params={},token_sha256='token')
            first=publish(root,b'state1',payload,info);self.assertIsNone(snapshot(root,'token'))
            info.update(global_step=2000,ready=True,actor_updates=1000);payload['version']=1000
            second=publish(root,b'state2',payload,info)
            self.assertEqual(current(root),second);self.assertEqual((first/'learner.msgpack').read_bytes(),b'state1')
            self.assertEqual(snapshot(root,'token')[0]['version'],1000)
    def test_inference_waiter_has_priority_over_another_gradient(self):
        gate=PriorityGate();started=threading.Event();finished=threading.Event()
        def infer():
            started.set()
            with gate.inference():finished.set()
        with gate.learning() as yes:
            self.assertTrue(yes);thread=threading.Thread(target=infer);thread.start();started.wait(1)
            for _ in range(100):
                if gate.waiters:break
                time.sleep(.001)
            with gate.learning() as allowed:self.assertFalse(allowed)
        thread.join(1);self.assertTrue(finished.is_set())
    def test_generation_echo_and_feature_request_do_not_execute_actor(self):
        class Fake:
            last={}
            def infer(self,obs,features_only=False):
                self.only=features_only;return np.zeros((50,7))
        fake=Fake();service=Service(fake)
        result=service.infer(dict(method='features',session='s',request_id=1,generation=8,observed_at=10,cam0=0,cam1=0,state=0,prompt=''))
        self.assertTrue(fake.only);self.assertEqual(result['generation'],8)

if __name__=='__main__':unittest.main()
