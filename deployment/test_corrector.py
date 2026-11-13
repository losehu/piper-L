"""No hardware: monitor alignment, recovery generations, service and normalization."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import tempfile
import threading
import unittest
from unittest import mock
import numpy as np
from corrector.contract import VERSION,action_normalize
from corrector.monitor import RobustGate,LatentMonitor
from corrector.serve import Service
from corrector.execution import ExecutionState
from rtc_piper.cli import config
from rtc_piper.profiles import select_model

ROOT=Path(__file__).resolve().parent


def gate_settings():
    return dict(window=30,warmup=3,persistence=2,cooldown_s=.3,on_mad=4.,off_mad=2.,floor=.3,mad_floor=.005,max_interpolation_gap_s=.2)


class MonitorTests(unittest.TestCase):
    def test_normalized_action_matches_openpi_delta_then_quantile_without_clipping(self):
        state=np.array([.2,.3,-.4,.5,-.6,.7,.03]);action=state+.1
        stats=dict(q01=[-.2]*6+[0],q99=[.2]*6+[.1])
        expected=np.r_[np.full(6,.1),.13]
        expected=(expected-np.array(stats['q01']))/(np.array(stats['q99'])-stats['q01']+1e-6)*2-1
        np.testing.assert_allclose(action_normalize(action,state,stats),expected,atol=1e-6)
        self.assertGreater(action_normalize(action,state,stats)[-1],1)

    def test_persistent_error_triggers_once_and_requires_release(self):
        gate=RobustGate(gate_settings())
        for i in range(8):self.assertFalse(gate.update(.1,i*.1)['event'])
        self.assertFalse(gate.update(1.5,.8)['event']);self.assertTrue(gate.update(1.5,.9)['event'])
        for i in range(10,20):self.assertFalse(gate.update(1.5,i*.1)['event'])
        self.assertFalse(gate.update(.1,2.)['latched'])

    def test_feature_interpolation_uses_capture_times_and_ignores_stale_messages(self):
        interval=5/30;monitor=LatentMonitor(lambda z,a:np.ones_like(z)*interval,5,gate_settings())
        for t in np.arange(0,2,.1):
            r=monitor.observe(np.ones((3,4))*t,np.zeros(7),float(t))
            self.assertFalse(r.get('event'))
            if 'error' in r:self.assertLess(r['error'],1e-5)
        self.assertGreater(monitor.stats['scores'],10)
        monitor.observe(np.zeros((3,4)),np.zeros(7),.1)
        self.assertEqual(monitor.stats['stale_observations'],1)

    def test_static_or_missing_images_do_not_get_random_anomaly_scores(self):
        monitor=LatentMonitor(lambda z,a:np.ones_like(z),5,gate_settings())
        for t in np.arange(0,1,.1):monitor.observe(np.zeros((3,4)),np.zeros(7),float(t))
        self.assertEqual(monitor.stats['scores'],0);self.assertGreater(monitor.stats['static_transitions'],0)
        monitor.observe(np.ones((3,4)),np.zeros(7),2.)
        self.assertGreater(monitor.stats['unaligned_predictions'],0)
        with self.assertRaises(ValueError):monitor.observe(np.zeros((3,4)),np.zeros(7),float('nan'))


class GenerationTests(unittest.TestCase):
    def setUp(self):
        settings=select_model(config(ROOT/'config.example.yaml'),'pi05',ROOT)
        self.initial=np.array([0,.1,-.2,0,.3,0,.02]);self.state=ExecutionState(ROOT,settings,self.initial,10.)

    def reply(self,generation=0,event=0,at=None,needs=False):
        return dict(generation=generation,event_id=event,event_at=at,needs_replan=needs,actions=np.tile(self.initial,(50,1)))

    def test_inflight_old_generation_cannot_repopulate_a_truncated_queue(self):
        self.assertTrue(self.state.publish(self.reply(),10.,0,10.1))
        self.state.apply_event(self.reply(1,1,10.2,True))
        self.assertIsNone(self.state.sample(10.25)[0])
        self.assertFalse(self.state.publish(self.reply(),10.1,0,10.3))
        self.assertIsNone(self.state.sample(10.3)[0])
        self.assertTrue(self.state.publish(self.reply(1,1,10.2),10.21,1,10.31))
        self.assertIsNotNone(self.state.sample(10.32)[0]);self.assertEqual(self.state.stats['recoveries'],1)

    def test_recovery_rejects_images_from_before_the_interrupt(self):
        self.state.apply_event(self.reply(1,1,10.2,True))
        self.assertFalse(self.state.publish(self.reply(1,1,10.2),10.19,1,10.3))
        self.state.apply_event(self.reply(0,0,None))
        self.assertEqual(self.state.generation,1)

    def test_command_alignment_uses_the_actually_sent_past_command(self):
        later=self.initial+.01;self.state.record_command(10.1,later)
        np.testing.assert_array_equal(self.state.command_at(10.05),self.initial)
        np.testing.assert_array_equal(self.state.command_at(10.15),later)


class ServiceTests(unittest.TestCase):
    def test_full_guides_only_one_recovery_and_monitor_mode_does_not_truncate(self):
        class FakePolicy:
            def __init__(self):self.calls=[]
            def infer(self,obs,**kwargs):
                self.calls.append(kwargs['enabled']);return np.zeros((50,7)),np.ones((3,4)),None
            def encode(self,obs):return np.ones((3,4))
            def predict(self,z,a):return np.ones_like(z)
        for mode in ('monitor','truncate','full'):
            policy=FakePolicy();service=Service(policy,dict(contract=dict(k_step=5),monitor_config=gate_settings()),dict(actions=dict(q01=[-1]*7,q99=[1]*7)))
            req=dict(request_id=1,session='test-session',mode=mode,method='infer',generation=0,observed_at=10.,
                cam0=np.zeros((2,2,3)),cam1=np.zeros((2,2,3)),state=np.zeros(7),prompt='task',executed_command=np.zeros(7))
            service.adopt(req['session'],mode)
            def event(z,a,t):
                service.monitor.last_event=dict(stamp=t,target=np.ones_like(z));return dict(event=True)
            service.monitor.observe=event
            result=service.infer(req)
            self.assertEqual(result['generation'],0 if mode=='monitor' else 1)
            self.assertEqual(result['needs_replan'],mode!='monitor')
            service.monitor.observe=lambda *args:dict(event=False)
            req.update(generation=result['generation'],observed_at=10.1,request_id=2)
            service.infer(req);req.update(observed_at=10.2,request_id=3);service.infer(req)
            self.assertEqual(policy.calls,[False,mode=='full',False])


class ServiceOwnershipTests(unittest.TestCase):
    def setup_state(self,root):
        args=['python','-u','-m','corrector.serve','--root',str(root),'--port','8002']
        (root/'logs').mkdir();(root/'logs/corrector_policy_state.json').write_text(json.dumps(dict(pid=12345,owner_root=str(root),port=8002,argv=args)))
        return args

    def test_connected_or_unknown_corrector_is_never_killed(self):
        import rtc_piper.model_services as service
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.setup_state(root)
            with mock.patch.object(service,'process_args',return_value=args),mock.patch.object(service,'connected',return_value=True),mock.patch.object(service.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'connected client'):service.stop_idle_corrector(root)
                kill.assert_not_called()
            with mock.patch.object(service,'process_args',return_value=['other-job']),mock.patch.object(service.os,'kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'identity changed'):service.stop_idle_corrector(root)
                kill.assert_not_called()

    def test_only_exact_owned_idle_corrector_pid_is_released(self):
        import rtc_piper.model_services as service
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);args=self.setup_state(root)
            with mock.patch.object(service,'process_args',side_effect=[args,[]]),mock.patch.object(service,'connected',return_value=False),mock.patch.object(service.os,'kill') as kill:
                service.stop_idle_corrector(root);kill.assert_called_once_with(12345,service.signal.SIGTERM)


class LoopbackConnectionTests(unittest.TestCase):
    def test_local_corrector_connects_even_when_desktop_proxies_reject_connections(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        from websockets.sync.server import serve
        from websockets.exceptions import ConnectionClosed
        from openpi_client import msgpack_numpy
        from corrector.client import Client,Session
        class RejectingProxy(BaseHTTPRequestHandler):
            def do_CONNECT(self):
                self.server.hits+=1;self.send_error(502,'Test proxy rejects local connections')
            def log_message(self,*args):pass
        metadata=dict(protocol=VERSION,training_updates=2500,source_checkpoint_index=2499,
                      action_horizon=50,physical_action_dim=7,warmed_up=True,robot_connected=False)
        def handler(ws):
            ws.send(msgpack_numpy.packb(metadata))
            try:ws.recv()
            except ConnectionClosed:pass
        with ThreadingHTTPServer(('127.0.0.1',0),RejectingProxy) as proxy, \
                serve(handler,'127.0.0.1',0) as server:
            proxy.hits=0
            threads=[threading.Thread(target=proxy.serve_forever,daemon=True),
                     threading.Thread(target=server.serve_forever,daemon=True)]
            for thread in threads:thread.start()
            url='http://127.0.0.1:%d'%proxy.server_port
            env={key:url for key in ('http_proxy','https_proxy','all_proxy','ws_proxy','wss_proxy')}
            env['no_proxy']=''
            try:
                with mock.patch.dict(os.environ,env,clear=True):
                    client=Client(server.socket.getsockname()[1],Session('full'))
                    try:self.assertEqual(client.metadata,metadata)
                    finally:client.close()
                self.assertEqual(proxy.hits,0)
            finally:
                server.shutdown();proxy.shutdown()
                for thread in threads:thread.join(2)


if __name__=='__main__':unittest.main()
