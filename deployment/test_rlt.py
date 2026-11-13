"""No robot: action contracts, replay provenance, explicit labels and RPC service."""
from pathlib import Path
import json,tempfile,unittest
from types import SimpleNamespace
import numpy as np
from rlt_piper.contract import SCHEMA,UPSTREAM,CHUNK,normalize,absolute,assets,digest
from rlt_piper.recording import Recorder,transitions
from rlt_piper.serve import Service

class ContractTests(unittest.TestCase):
    def test_delta_joints_absolute_gripper_roundtrip_including_zero_rows(self):
        stats=dict(q01=[-1]*6+[0],q99=[1]*6+[.1]);state=np.array([.2,.3,-.4,.1,.2,.3,.02])
        values=np.zeros((18,7));values[:,6]=.03
        normalized=normalize(values,state,stats)
        np.testing.assert_allclose(absolute(normalized,state,stats),values,atol=1e-6)
        np.testing.assert_allclose(normalized[:,:6],np.tile(-state[:6],(18,1)),atol=1e-6)
        self.assertAlmostEqual(normalized[0,6],-.4,places=4)
    def test_actor_and_missing_token_never_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with self.assertRaisesRegex(RuntimeError,'not trained'):assets(root)
            (root/'token.msgpack').write_bytes(b'unit-test-token');(root/'norm_stats.json').write_text('{}')
            meta=dict(schema=SCHEMA,upstream=UPSTREAM,base_step=2500,base_frozen=True,trained=True,
                      token_sha256=digest(root/'token.msgpack'),norm_sha256=digest(root/'norm_stats.json'))
            (root/'token.json').write_text(json.dumps(meta));assets(root)
            with self.assertRaisesRegex(RuntimeError,'not trained'):assets(root,actor=True)
            (root/'actor.pkl').write_bytes(b'unit-test-actor')
            (root/'actor.json').write_text(json.dumps(dict(schema=SCHEMA,chunk_len=CHUNK,token_sha256=meta['token_sha256'],
                actor_updates=0,actor_sha256=digest(root/'actor.pkl'))))
            with self.assertRaisesRegex(RuntimeError,'untrained'):assets(root,actor=True)
            (root/'token.msgpack').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'identity'):assets(root)

class ReplayTests(unittest.TestCase):
    def example(self,root,source='rlt',generation=0,label=1):
        cfg=dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05)
        r=Recorder(root,'online','token',cfg)
        for i in range(8):
            t=10.+i*.1;obs=SimpleNamespace(t=t)
            r.plan(obs,dict(generation=generation if i<4 else 0,rlt=dict(z_rl=np.ones(512)*i,proprio=np.zeros(7),
                ref_chunk=np.ones((18,7))*.02,proposed_chunk=np.ones((18,7))*.9,source=source,actor_version=4)),t-.01)
        for t in np.arange(9.99,10.851,1/120):r.command(t,np.ones(7)*.01)
        r.label=label;r.label_at=10.85
        terminal=dict(t=10.85,z_rl=np.ones(512)*10,proprio=np.zeros(7),ref_chunk=np.ones((18,7))*.02)
        return r.save(terminal)
    def test_replay_uses_actual_transmitted_commands_and_terminal_padding_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.example(root);rows=transitions(p,'token');self.assertTrue(rows)
            for row in rows:np.testing.assert_allclose(row['action_chunk'],.01)
            tail=[r for r in rows if r['done']];self.assertTrue(tail)
            self.assertTrue(any(r['valid_steps']<CHUNK for r in tail))
            for row in tail:self.assertEqual(row['rewards'][row['valid_steps']-1],1)
    def test_abort_or_wrong_token_is_not_training_data(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.example(root,label=None)
            self.assertEqual(transitions(p,'token'),[])
            self.assertEqual(transitions(p,'different-token'),[])
    def test_post_stop_observation_delay_does_not_erase_terminal_reward(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.example(root)
            with np.load(p/'trace.npz') as d:values={k:d[k] for k in d.files}
            values['terminal_t']=np.array(11.3)
            np.savez_compressed(p/'trace.npz',**values)
            rows=transitions(p,'token')
            self.assertTrue(any(r['done'] and r['rewards'].sum()==1 for r in rows))
    def test_corrector_sources_are_excluded_from_this_no_corrector_route(self):
        with tempfile.TemporaryDirectory() as root:self.assertEqual(transitions(self.example(root,source='corrector'),'token'),[])
    def test_label_is_explicit_and_only_first_outcome_wins(self):
        with tempfile.TemporaryDirectory() as root:
            r=Recorder(root,'collect','token',dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05))
            guard=SimpleNamespace(stop=SimpleNamespace(value=0));r.outcome(0,guard);r.outcome(1,guard)
            self.assertEqual(r.label,0);self.assertEqual(guard.stop.value,1)
    def test_an_existing_emergency_stop_cannot_be_relabelled_as_success(self):
        with tempfile.TemporaryDirectory() as root:
            r=Recorder(root,'collect','token',dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05))
            guard=SimpleNamespace(stop=SimpleNamespace(value=1));r.outcome(1,guard)
            self.assertIsNone(r.label)

class ServiceTests(unittest.TestCase):
    def test_no_corrector_or_actor_is_required_for_base_collection_service(self):
        class Fake:
            last=dict(source='base',actor_version=0,z_rl=np.ones(512),proprio=np.zeros(7),ref_chunk=np.zeros((18,7)))
            def infer(self,obs):return np.zeros((50,7))
        s=Service(Fake());request=dict(method='infer',session='unit',request_id=1,observed_at=10.,cam0=0,cam1=0,state=np.zeros(7),prompt='test')
        out=s.infer(request);self.assertEqual(out['protocol'],SCHEMA);self.assertEqual(out['rlt']['source'],'base')
        request['observed_at']=float('nan')
        with self.assertRaises(ValueError):s.infer(request)

if __name__=='__main__':unittest.main()
