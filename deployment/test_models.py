"""CPU/mock contracts for model selection and slow VPP2 planning. No GPU/CAN."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from contextlib import contextmanager,nullcontext
from http.server import ThreadingHTTPServer
from unittest import mock
import numpy as np

from rtc_piper.cli import config,ensure_policy
from rtc_piper.profiles import select_model
from rtc_piper.observations import Observation
from rtc_piper.policy import validate_actions
from rtc_piper.vpp2_client import Vpp2Client,PROTOCOL,PROMPT
from rtc_piper.vpp2_control import run_vpp2

ROOT=Path(__file__).resolve().parent


class SelectionTests(unittest.TestCase):
    def test_models_keep_arm_protocol_but_select_different_action_shapes(self):
        cfg=config(ROOT/'config.yaml')
        pi=select_model(cfg,'pi05',ROOT);vpp=select_model(cfg,'vpp2',ROOT)
        self.assertEqual((pi['model_step'],pi['action_horizon'],pi['execution_mode']),(2500,50,'rtc'))
        self.assertEqual((vpp['model_step'],vpp['action_horizon'],vpp['execution_mode']),(2250,32,'plan_execute'))
        for key in ['channel','firmware_driver','lock_file','policy_command_lower_rad','policy_command_upper_rad']:
            self.assertEqual(pi[key],vpp[key])
        self.assertEqual(pi['policy_timeout_s'],.8)
        self.assertGreater(vpp['policy_timeout_s'],vpp['request_timeout_s']+32/30)

    def test_shape_is_strict_per_backend_and_outputs_keep_firmware_clipping(self):
        cfg=select_model(config(ROOT/'config.yaml'),'vpp2',ROOT)
        actions=np.zeros((32,7));actions[:,5]=np.radians(-190);actions[:,6]=.12
        out=validate_actions(actions,cfg)
        np.testing.assert_allclose(np.degrees(out[:,5]),-170)
        np.testing.assert_allclose(out[:,6],.1)
        with self.assertRaises(RuntimeError):validate_actions(np.zeros((50,7)),cfg)
        with self.assertRaises(RuntimeError):validate_actions(np.full((32,7),np.nan),cfg)

    def test_compile_only_changes_vpp2_compute_and_startup_budget(self):
        cfg=config(ROOT/'config.yaml');normal=select_model(cfg,'vpp2',ROOT)
        compiled=select_model(cfg,'vpp2',ROOT,compile_vpp2=True)
        self.assertTrue(compiled['compile_vpp2']);self.assertFalse(normal['compile_vpp2'])
        for key in ['fps','action_horizon','request_timeout_s','policy_command_lower_rad']:
            self.assertEqual(normal[key],compiled[key])
        with self.assertRaises(ValueError):select_model(cfg,'pi05',ROOT,compile_vpp2=True)

    def test_vpp2_defaults_to_five_steps_with_explicit_ten_step_reference(self):
        cfg=config(ROOT/'config.yaml');cfg['vpp2']={}
        self.assertEqual(select_model(cfg,'vpp2',ROOT)['denoising_steps'],5)
        cfg['vpp2']['denoising_steps']=10
        self.assertEqual(select_model(cfg,'vpp2',ROOT)['denoising_steps'],10)
        cfg['vpp2']['denoising_steps']=0
        with self.assertRaises(ValueError):select_model(cfg,'vpp2',ROOT)

    def test_missing_vpp2_assets_never_stop_existing_model(self):
        cfg=select_model(config(ROOT/'config.yaml'),'vpp2',ROOT)
        with tempfile.TemporaryDirectory() as folder:
            cfg['vpp2_root']=folder
            with mock.patch('rtc_piper.hardware.follower_lock',return_value=nullcontext()), \
                    mock.patch('rtc_piper.model_services.stop_idle_other') as stop:
                with self.assertRaisesRegex(RuntimeError,'missing'):ensure_policy(cfg)
        stop.assert_not_called()


@contextmanager
def fake_service():
    spec=importlib.util.spec_from_file_location('vpp2_mock_service',ROOT/'vpp2_runtime/server.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Handler(module.Handler):
        def log_message(self,*_):pass
        def _send(self,status,payload):
            if status==200 and 'actions' in payload and self.server.policy.bad_request_id:
                payload=dict(payload,request_id='old-session')
            return super()._send(status,payload)
    class Policy:
        step=2250;warmup_ms=1.;compiled_action=False;denoising_steps=5
        lock=threading.Lock();override={};captured=None;bad_request_id=False
        def infer_prepared(self,a,b,state,seed=42):
            self.captured=(a,b,state)
            values=np.tile([.11,.22,-.33,.44,-.55,.66,.033],(32,1))
            result=dict(protocol=PROTOCOL,ok=True,checkpoint_step=2250,actions=values.tolist(),
                denoising_steps=5,
                action_units='rad,m',action_type='absolute_joint_targets',action_rate_hz=30,
                action_horizon=32,normalized_action=[],inference_ms=1)
            result.update(self.override)
            return result
    service=ThreadingHTTPServer(('127.0.0.1',0),Handler);service.policy=Policy()
    worker=threading.Thread(target=service.serve_forever,daemon=True);worker.start()
    cfg=select_model(config(ROOT/'config.yaml'),'vpp2',ROOT);cfg['policy_port']=service.server_port
    try:yield cfg,service.policy
    finally:service.shutdown();service.server_close();worker.join(1)


class HttpContractTests(unittest.TestCase):
    def observation(self):
        a=np.zeros((480,640,3),np.uint8);a[:,:,0]=200
        b=np.zeros_like(a);b[:,:,2]=180
        return Observation(10,np.array([1,.4,-.4,.1,.3,-.2,.04]),dict(cam0=a,cam1=b),np.full(4,10))

    def test_real_http_transport_preserves_camera_order_and_absolute_actions(self):
        with fake_service() as (cfg,policy):
            client=Vpp2Client(cfg);out=client.infer(self.observation(),PROMPT);client.close()
            np.testing.assert_allclose(out[0],[.11,.22,-.33,.44,-.55,.66,.033])
            a,b,state=policy.captured
            self.assertEqual(a.shape,(224,224,3));self.assertEqual(b.shape,(224,224,3))
            self.assertTrue(np.all(a[:28]==0));self.assertGreater(a[100,100,0],180)
            self.assertGreater(b[100,100,2],160)
            np.testing.assert_allclose(state,self.observation().state)

    def test_old_identity_and_unwarmed_service_are_rejected(self):
        with fake_service() as (cfg,policy):
            policy.step=2500
            with self.assertRaises(RuntimeError):Vpp2Client(cfg)
            policy.step=2250;policy.warmup_ms=None
            with self.assertRaises(RuntimeError):Vpp2Client(cfg)

    def test_compile_mode_mismatch_is_rejected(self):
        with fake_service() as (cfg,policy):
            cfg['compile_vpp2']=True
            with self.assertRaises(RuntimeError):Vpp2Client(cfg)

    def test_old_ten_step_service_or_response_is_rejected(self):
        with fake_service() as (cfg,policy):
            policy.denoising_steps=10
            with self.assertRaises(RuntimeError):Vpp2Client(cfg)
            policy.denoising_steps=5;client=Vpp2Client(cfg)
            policy.override=dict(denoising_steps=10)
            with self.assertRaisesRegex(RuntimeError,'mismatch'):client.infer(self.observation(),PROMPT)
            client.close()

    def test_invalid_units_or_request_identity_do_not_become_actions(self):
        with fake_service() as (cfg,policy):
            client=Vpp2Client(cfg)
            for invalid in [dict(action_units='deg,mm'),dict(action_type='delta')]:
                policy.override=invalid
                with self.assertRaisesRegex(RuntimeError,'mismatch'):client.infer(self.observation(),PROMPT)
            policy.override={};policy.bad_request_id=True
            with self.assertRaisesRegex(RuntimeError,'mismatch'):client.infer(self.observation(),PROMPT)
            client.close()

    def test_closed_client_never_sends_a_new_request(self):
        with fake_service() as (cfg,policy):
            client=Vpp2Client(cfg);client.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):client.infer(self.observation(),PROMPT)
            self.assertIsNone(policy.captured)


class SlowPlanningTests(unittest.TestCase):
    def invoke(self,delay,stop_at=14.5,observation_lag=0):
        import rtc_piper.vpp2_control as control
        clock=[10.];state=np.r_[np.zeros(6),.02];commands=[];jobs=[]
        def check():
            if clock[0]>=stop_at:raise RuntimeError('operator stop')
        guard=types.SimpleNamespace(check=check,stop=types.SimpleNamespace(value=0),
            running=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=0),
            startup_deadline=types.SimpleNamespace(value=0))
        def joints(q):state[:6]=q;commands.append((clock[0],np.array(q)))
        def grip(value,force):state[6]=value
        arm=types.SimpleNamespace(joints=joints,grip=grip,status=lambda **_:None)
        bank=types.SimpleNamespace(live=lambda:(state.copy(),np.full(4,clock[0])),
            observation=lambda:Observation(clock[0]-observation_lag,state.copy(),{},np.full(4,clock[0]-observation_lag)))
        actions=np.tile(state,(32,1));actions[:,0]=np.linspace(0,.1,32)
        class Job:
            def __init__(self,client,observation,prompt):
                self.observation=observation;self.actions=actions;self.error=None;self.started=clock[0]
                self.done=types.SimpleNamespace(is_set=lambda:clock[0]>=self.started+delay)
                self.thread=mock.Mock();jobs.append(self)
                self.thread.join.side_effect=lambda *_:self_test.assertEqual(guard.stop.value,1)
        self_test=self
        cfg=select_model(config(ROOT/'config.yaml'),'vpp2',ROOT)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(control,'InferenceJob',Job), \
                mock.patch.object(control.time,'monotonic',side_effect=lambda:clock[0]), \
                mock.patch.object(control.time,'sleep',side_effect=lambda s:clock.__setitem__(0,clock[0]+max(s,.0001))), \
                mock.patch('builtins.print'):
            with self.assertRaises(RuntimeError) as error:
                run_vpp2(folder,arm,types.SimpleNamespace(check=lambda:None),bank,guard,
                    types.SimpleNamespace(metadata={'model':'vpp2_piper_2250'}),state.copy(),9.9,cfg)
            report=json.loads((Path(folder)/'logs/last_policy_run.json').read_text())
        return str(error.exception),commands,jobs,report,guard

    def test_two_second_inference_keeps_120hz_hold_without_executing_old_actions(self):
        error,commands,jobs,report,guard=self.invoke(2.)
        self.assertIn('operator stop',error);self.assertEqual(report['stats']['accepted_chunks'],1)
        first_reply=jobs[0].started+2
        before=[q for t,q in commands if t<first_reply]
        self.assertGreater(len(before),200)
        np.testing.assert_allclose(before,0)
        self.assertLess(np.max(np.diff([t for t,q in commands])),.011)
        self.assertTrue(any(q[0]>.04 for t,q in commands if t>first_reply))
        self.assertEqual(guard.stop.value,1)

    def test_inference_timeout_stops_before_any_new_plan_is_executed(self):
        error,commands,jobs,report,guard=self.invoke(20.,stop_at=40.)
        self.assertIn('inference timed out',error)
        self.assertEqual(report['stats']['accepted_chunks'],0)
        np.testing.assert_allclose([q for t,q in commands],0)
        self.assertEqual(guard.stop.value,1)

    def test_next_request_waits_for_whole_chunk_and_physical_stop(self):
        error,commands,jobs,report,guard=self.invoke(.4,stop_at=17.)
        self.assertIn('operator stop',error);self.assertGreaterEqual(len(jobs),2)
        events=report['recent_events'];playing=None;completed=None;stopped=None
        for event in events:
            if event['stage']=='play':playing=event['t'];completed=None;stopped=None
            elif event['stage']=='chunk_complete':
                self.assertIsNotNone(playing)
                self.assertGreaterEqual(event['t']-playing,32/30)
                completed=event['t'];playing=None
            elif event['stage']=='stopped':stopped=event['t']
            elif event['stage']=='infer':
                self.assertIsNone(playing,'Inference must not overlap chunk playback')
                self.assertIsNotNone(stopped)
                self.assertGreater(event['observed_at'],stopped)
                if completed is not None:self.assertGreater(stopped,completed)
        self.assertGreaterEqual(report['stats']['completed_chunks'],1)

    def test_frames_from_before_stopping_are_not_used_for_next_inference(self):
        error,commands,jobs,report,guard=self.invoke(.4,stop_at=16.,observation_lag=.12)
        stopped=[e['t'] for e in report['recent_events'] if e['stage']=='stopped']
        self.assertGreaterEqual(len(jobs),2)
        for job,stamp in zip(jobs,stopped):
            self.assertGreater(job.observation.t,stamp)
            self.assertTrue(np.all(job.observation.state_times>stamp))
            self.assertGreater(job.started-stamp,.12)


class ServiceOwnershipTests(unittest.TestCase):
    def test_unknown_pid_is_not_killed(self):
        from rtc_piper.model_services import stop_idle_other
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            (root/'logs/policy_state.json').write_text(json.dumps({'pid':123}))
            with mock.patch('rtc_piper.model_services.process_args',return_value=['python','other_job.py']), \
                    mock.patch('rtc_piper.model_services.os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'identity'):stop_idle_other(root,'vpp2')
            kill.assert_not_called()

    def test_connected_model_client_is_not_interrupted(self):
        from rtc_piper.model_services import stop_idle_other
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'logs').mkdir()
            (root/'logs/policy_state.json').write_text(json.dumps({'pid':123}))
            argv=['python','-m','rtc_piper.serve_policy','--root',str(root)]
            with mock.patch('rtc_piper.model_services.process_args',return_value=argv), \
                    mock.patch('rtc_piper.model_services.connected',return_value=True), \
                    mock.patch('rtc_piper.model_services.os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'connected'):stop_idle_other(root,'vpp2')
            kill.assert_not_called()


if __name__=='__main__':unittest.main()
