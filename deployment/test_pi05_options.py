"""Pi05 checkpoint/ablation contracts, with synthetic state and no devices."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest import mock
import numpy as np
from rtc_piper.cli import config
from rtc_piper.profiles import select_model
from rtc_piper.pi05_options import checkpoint_settings
from rtc_piper.policy import Timeline,CommandSmoother
from rtc_piper.pi05_sync import SyncChunks
from rtc_piper.observations import Observation
from rtc_piper.model_services import prepare_pi05

ROOT=Path(__file__).resolve().parent
def selected(**kwargs):return select_model(config(ROOT/'config.example.yaml'),'pi05',ROOT,**kwargs)


class OptionsTests(unittest.TestCase):
    def test_pi05_smoothing_tuning_is_shared_with_corrector_but_isolated_from_vpp2(self):
        base=config(ROOT/'config.example.yaml');unchanged=json.dumps(base,sort_keys=True)
        pi=select_model(base,'pi05',ROOT)
        vpp=select_model(base,'vpp2',ROOT)
        self.assertEqual((pi['chunk_smoothing_steps'],pi['command_filter_tau_s']),(12,.05))
        self.assertEqual((vpp['chunk_smoothing_steps'],vpp['command_filter_tau_s']),(10,.04))
        self.assertEqual(json.dumps(base,sort_keys=True),unchanged)
        disabled=select_model(base,'pi05',ROOT,no_smoothing=True)
        self.assertFalse(disabled['smoothing_enabled'])
        legacy=dict(base);legacy.pop('pi05')
        old=select_model(legacy,'pi05',ROOT)
        self.assertEqual((old['chunk_smoothing_steps'],old['command_filter_tau_s']),(10,.04))

    def test_pi05_tuning_rejects_invalid_values_before_hardware(self):
        base=config(ROOT/'config.example.yaml')
        for key,values in [('chunk_smoothing_steps',[True,0,19,2.5]),
                           ('command_filter_tau_s',[True,0,.3,float('nan'),float('inf'),'0.05'])]:
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,'pi05'):
                    select_model(dict(base,pi05={key:value}),'pi05',ROOT)

    def test_explicit1500_and_default2500_have_distinct_verified_identities(self):
        default=selected();old=selected(pi05_step=1500)
        self.assertEqual((default['model_step'],default['checkpoint_index']),(2500,2499))
        self.assertEqual((old['model_step'],old['checkpoint_index']),(1500,1500))
        self.assertEqual(old['checkpoint'],'checkpoints/pi05_piper_1500_bf16')
        bad=dict(old,checkpoint_index=1499)
        with self.assertRaisesRegex(ValueError,'identity'):checkpoint_settings(bad)

    def test_ablation_flags_are_composable_and_do_not_mutate_defaults_or_vpp2(self):
        base=config(ROOT/'config.example.yaml');unchanged=json.dumps(base,sort_keys=True)
        cfg=select_model(base,'pi05',ROOT,pi05_step=1500,no_smoothing=True,no_rtc=True)
        self.assertEqual((cfg['model_step'],cfg['execute_horizon'],cfg['execution_mode']),(1500,50,'sync_chunks'))
        self.assertFalse(cfg['smoothing_enabled']);self.assertEqual(json.dumps(base,sort_keys=True),unchanged)
        self.assertEqual(selected()['execution_mode'],'rtc');self.assertNotIn('smoothing_enabled',selected())
        with self.assertRaisesRegex(ValueError,'pi05 only'):select_model(base,'vpp2',ROOT,no_rtc=True)

    def test_no_smoothing_bypasses_both_bridge_and_filter_but_keeps_limits(self):
        cfg=selected(no_smoothing=True);initial=np.array([0,.1,-.2,0,.3,0,.02])
        actions=np.tile(initial,(50,1));actions[:,0]=.2
        timeline=Timeline(ROOT,cfg,10,initial_action=initial)
        report=timeline.enqueue(actions,10,10.08)
        self.assertEqual(report['boundary_blended_frames'],0);self.assertAlmostEqual(timeline.target(10.09)[0],.2)
        smoother=CommandSmoother(initial,cfg);target=initial.copy();target[0]=.05
        self.assertAlmostEqual(smoother.step(target,.1)[0],.05)
        target[0]=9;out=smoother.step(target,.01)
        self.assertAlmostEqual(out[0]-.05,np.radians(cfg['policy_joint_speed_deg_s'][0])*.01)
        with self.assertRaisesRegex(RuntimeError,'100ms'):smoother.step(target,.101)

    def test1500_client_rejects_a2500_service(self):
        from rtc_piper.policy import Client
        from openpi_client import msgpack_numpy
        cfg=selected(pi05_step=1500)
        meta=dict(model='pi05_piper_grasp_1500',training_updates=1500,source_checkpoint_index=1500,
                  action_horizon=50,physical_action_dim=7,fps=30,warmed_up=True)
        for step in [2500,1500]:
            ws=mock.Mock();ws.recv.return_value=msgpack_numpy.packb(dict(meta,training_updates=step))
            with mock.patch('websockets.sync.client.connect',return_value=ws):
                if step==2500:
                    with self.assertRaisesRegex(RuntimeError,'identity'):Client(cfg)
                else:Client(cfg).close()


class SwitchingTests(unittest.TestCase):
    def test_only_different_idle_policy_is_selected_for_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'logs').mkdir();cfg=selected(pi05_step=1500)
            path=root/cfg['checkpoint'];(path/'params').mkdir(parents=True);(path/'assets').mkdir()
            state=root/'logs/policy_state.json'
            for old_step,old_index,expected in [(1500,1500,False),(2500,2499,True)]:
                state.write_text(json.dumps(dict(pid=123,metadata=dict(training_updates=old_step,source_checkpoint_index=old_index))))
                with mock.patch('rtc_piper.model_services.process_args',return_value=['owned']), \
                     mock.patch('rtc_piper.model_services.stop_idle_other') as stop:
                    prepare_pi05(root,cfg);self.assertEqual(stop.called,expected)
            import shutil
            shutil.rmtree(path)
            with mock.patch('rtc_piper.model_services.stop_idle_other') as stop:
                with self.assertRaisesRegex(RuntimeError,'missing'):prepare_pi05(root,cfg)
                stop.assert_not_called()


class SynchronousTests(unittest.TestCase):
    def setUp(self):
        self.cfg=selected(no_smoothing=True,no_rtc=True)
        self.state=np.array([0,.1,-.2,0,.3,0,.02]);self.calls=[];self.jobs=[]
        self.actions=np.tile(self.state,(50,1));self.actions[:,0]=np.arange(50)*.002
        def factory(client,obs,prompt):
            self.calls.append(obs.t);done=threading.Event();done.set()
            job=SimpleNamespace(observation=obs,done=done,error=None,actions=self.actions.copy())
            self.jobs.append(job);return job
        self.plan=SyncChunks(None,self.state,9,self.cfg,10,job_factory=factory)
        self.plan.stable=SimpleNamespace(update=lambda *args:True)

    def tick(self,t,observed=None,state=None):
        state=self.state if state is None else state
        observed=t if observed is None else observed
        obs=Observation(observed,state.copy(),{},np.full(4,observed))
        return self.plan.target(t,state,np.full(4,t),obs)

    def boot(self):
        self.tick(10);self.tick(10.01,10)
        self.assertEqual(self.calls,[])
        self.tick(10.02);self.assertEqual(self.plan.stage,'infer')
        self.tick(10.03);self.assertEqual(self.plan.stage,'play')

    def test_full50_frames_then_fresh_observation_without_prefetch(self):
        self.boot();begun=self.plan.play_started
        for i in range(1,200):self.tick(begun+i/120)
        self.assertEqual(len(self.calls),1);self.assertEqual(self.plan.stage,'play')
        np.testing.assert_allclose(self.tick(begun+50/30+.001),self.actions[-1])
        self.assertEqual(self.plan.completed,1);self.assertEqual(len(self.calls),1)
        self.plan.stable=SimpleNamespace(update=lambda *args:True)
        stopped=begun+50/30+.01;self.tick(stopped)
        self.tick(stopped+.01,stopped-.1);self.assertEqual(len(self.calls),1)
        self.tick(stopped+.02);self.assertEqual(len(self.calls),2)

    def test_timeout_failure_and_external_motion_never_publish_a_plan(self):
        self.tick(10);self.tick(10.02)
        self.jobs[-1].done.clear()
        with self.assertRaisesRegex(RuntimeError,'timed out'):self.tick(13)
        self.assertEqual(self.plan.accepted,0)
        self.jobs[-1].done.set();moved=self.state.copy();moved[0]+=.1
        self.tick(10.03,state=moved)
        self.assertEqual(self.plan.stage,'settle');self.assertEqual(self.plan.discarded,1)
        self.assertEqual(self.plan.accepted,0)


if __name__=='__main__':unittest.main()
