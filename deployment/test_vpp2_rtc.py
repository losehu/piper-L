"""VPP2-only capture-time, continuity, underrun and background-producer checks."""
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock
import numpy as np
from rtc_piper.cli import config
from rtc_piper.profiles import select_model
from rtc_piper.observations import Observation
from rtc_piper.vpp2_rtc import CaptureTimeline,Producer

ROOT=Path(__file__).resolve().parent


def settings():
    base=config(ROOT/'config.example.yaml');base['vpp2']['inference']='h100_async_rtc'
    return select_model(base,'vpp2',ROOT)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.cfg=settings();self.initial=np.array([0,.1,-.2,0,.3,0,.02])
        self.actions=np.tile(self.initial,(32,1));self.actions[:,0]+=np.linspace(0,.2,32)
        self.timeline=CaptureTimeline(self.initial,self.cfg)

    def test_bootstrap_preserves_capture_stamp_and_moving_updates_skip_past_samples(self):
        first=self.timeline.publish(self.actions,10.,10.4)
        self.assertEqual(first['captured_at'],10.);self.assertEqual(first['scheduled_start'],10.4)
        np.testing.assert_allclose(self.timeline.sample(10.4),self.initial)
        second=self.timeline.publish(self.actions,10.41,10.83)
        self.assertEqual(second['captured_at'],10.41);self.assertEqual(second['scheduled_start'],10.41)
        self.assertEqual(second['skipped_prefix_frames'],12)
        self.assertAlmostEqual(second['valid_until'],10.41+32/30)

    def test_new_block_matches_old_position_and_tangent_on_arrival(self):
        self.timeline.publish(self.actions,10.,10.4);old=self.timeline.window.evaluate(10.83)
        changed=self.actions.copy();changed[:,0]+=.4
        self.timeline.publish(changed,10.41,10.83);new=self.timeline.window.evaluate(10.83)
        np.testing.assert_allclose(old[0],new[0],atol=1e-10)
        np.testing.assert_allclose(old[1],new[1],atol=1e-10)
        expected=self.timeline.window.base(11.2)[0]
        np.testing.assert_allclose(self.timeline.sample(11.2),expected)

    def test_no_old_window_extension_or_relabelling_after_expiry(self):
        self.timeline.publish(self.actions,10.,10.4);end=self.timeline.window.end
        self.assertIsNone(self.timeline.sample(end))
        with self.assertRaisesRegex(RuntimeError,'exhausted before replacement'):
            self.timeline.publish(self.actions,end-.2,end+.1)
        self.assertEqual(self.timeline.window.end,end)

    def test_wrong_shape_nan_old_capture_and_expired_response_are_rejected(self):
        with self.assertRaises(RuntimeError):self.timeline.publish(np.zeros((50,7)),10.,10.4)
        with self.assertRaises(RuntimeError):self.timeline.publish(np.full((32,7),np.nan),10.,10.4)
        with self.assertRaisesRegex(RuntimeError,'prediction window'):self.timeline.publish(self.actions,10.,11.2)
        self.timeline.publish(self.actions,10.,10.4)
        with self.assertRaisesRegex(RuntimeError,'did not advance'):self.timeline.publish(self.actions,10.,10.8)

    def test_pi05_profile_is_unchanged_when_vpp2_remote_is_selected(self):
        base=config(ROOT/'config.example.yaml');before=select_model(base,'pi05',ROOT)
        base['vpp2'].update(inference='h100_async_rtc',torch_compile=True,remote_port=21969)
        after=select_model(base,'pi05',ROOT)
        for key in ['model_step','checkpoint','action_horizon','execute_horizon','execution_mode',
                    'policy_timeout_s','chunk_smoothing_steps','command_filter_tau_s','policy_joint_speed_deg_s']:
            self.assertEqual(before[key],after[key])


class ProducerTests(unittest.TestCase):
    def run_producer(self,late=False,warmups=0):
        cfg=settings();cfg['vpp2_max_inflight']=1;cfg['vpp2_startup_warmups']=warmups
        clock=[10.];state=np.array([0,.1,-.2,0,.3,0,.02]);captures=[]
        guard=SimpleNamespace(stop=SimpleNamespace(value=0),policy_time=SimpleNamespace(value=0))
        timeline=CaptureTimeline(state,cfg)
        def observation():
            captures.append(clock[0]);return Observation(clock[0],state.copy(),{},np.full(4,clock[0]))
        def infer(obs,prompt):
            clock[0]+=1.2 if late and len(captures)==2 else .4
            if not late and len(captures)==5:guard.stop.value=1
            return np.tile(state,(32,1))
        producer=Producer(SimpleNamespace(infer=infer),observation,lambda:state,timeline,guard,cfg)
        with mock.patch('rtc_piper.vpp2_rtc.time.monotonic',side_effect=lambda:clock[0]):producer.run()
        return producer,timeline,guard,captures

    def test_requests_continue_before_previous_chunk_finishes(self):
        producer,timeline,guard,captures=self.run_producer()
        self.assertIsNone(producer.error);self.assertEqual(producer.stats['accepted_chunks'],4)
        np.testing.assert_allclose(np.diff(captures),.4)
        self.assertTrue(all(e['age_s']<32/30 for e in producer.events))

    def test_late_result_latches_stop_without_publishing_invalid_actions(self):
        producer,timeline,guard,captures=self.run_producer(late=True)
        self.assertIsNotNone(producer.error);self.assertEqual(guard.stop.value,1)
        self.assertEqual(timeline.accepted,1)

    def test_startup_warmups_finish_while_holding_before_first_motion_window(self):
        producer,timeline,guard,captures=self.run_producer(warmups=2)
        self.assertEqual(producer.stats['startup_warmup_calls'],2)
        self.assertAlmostEqual(producer.events[0]['captured_at'],10.8)
        self.assertAlmostEqual(producer.events[0]['scheduled_start'],11.2)

    def test_two_inflight_requests_discard_old_out_of_order_reply(self):
        cfg=settings();cfg['vpp2_startup_warmups']=0
        state=np.array([0,.1,-.2,0,.3,0,.02]);guard=SimpleNamespace(stop=SimpleNamespace(value=0),policy_time=SimpleNamespace(value=0))
        timeline=CaptureTimeline(state,cfg);counter=[0];lock=threading.Lock()
        def observation():
            now=time.monotonic();return Observation(now,state.copy(),{},np.full(4,now))
        def infer(obs,prompt):
            with lock:counter[0]+=1;index=counter[0]
            time.sleep(.08 if index==2 else .01)
            return np.tile(state,(32,1))
        producer=Producer(SimpleNamespace(infer=infer),observation,lambda:state,timeline,guard,cfg)
        producer.thread.start();until=time.monotonic()+3
        try:
            while producer.stats['accepted_chunks']<12 and time.monotonic()<until and not producer.error:time.sleep(.005)
        finally:guard.stop.value=1;producer.thread.join(.5)
        self.assertIsNone(producer.error);self.assertEqual(producer.stats['peak_inflight'],2)
        self.assertGreaterEqual(producer.stats['discarded_out_of_order'],1)
        captures=[e['captured_at'] for e in producer.events]
        self.assertTrue(np.all(np.diff(captures)>0));self.assertFalse(producer.thread.is_alive())

    def test_second_connection_starts_half_an_rtt_after_first_moving_request(self):
        cfg=settings();cfg['vpp2_startup_warmups']=0
        state=np.array([0,.1,-.2,0,.3,0,.02]);starts=[];lock=threading.Lock()
        guard=SimpleNamespace(stop=SimpleNamespace(value=0),policy_time=SimpleNamespace(value=0))
        timeline=CaptureTimeline(state,cfg)
        def observation():
            now=time.monotonic();return Observation(now,state.copy(),{},np.full(4,now))
        def infer(obs,prompt):
            with lock:starts.append(time.monotonic())
            time.sleep(.1)
            return np.tile(state,(32,1))
        producer=Producer(SimpleNamespace(infer=infer),observation,lambda:state,timeline,guard,cfg)
        producer.thread.start();until=time.monotonic()+3
        try:
            while len(starts)<10 and time.monotonic()<until and not producer.error:time.sleep(.005)
        finally:guard.stop.value=1;producer.thread.join(.5)
        self.assertIsNone(producer.error);self.assertGreaterEqual(len(starts),10)
        self.assertGreater(starts[2]-starts[1],.045)
        self.assertLess(starts[2]-starts[1],.09)
        self.assertAlmostEqual(producer.stats['startup_pipeline_offset_s'],.05,delta=.015)
        self.assertTrue(np.all(np.diff(starts[1:])>.045))


if __name__=='__main__':unittest.main()
