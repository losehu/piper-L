"""No-device regression checks, including the verified collector's CAN contract."""
import json
import math
from pathlib import Path
import socket
import struct
import tempfile
from contextlib import ExitStack,nullcontext
import types
import unittest
from unittest import mock

import numpy as np

from rtc_piper.cli import config
from rtc_piper.control import choose_start,trajectory,Settling,check_state
from rtc_piper.hardware import gated_send,JointCommunicationFault,SdkArm,apply_hardware_limits,read_hardware_limits
from rtc_piper.observations import CaptureBank,NotReady
from rtc_piper.policy import Timeline,validate_actions,limited_target,Producer,ActionRejection,CommandSmoother
from rtc_piper.smoothing import CriticallyDampedFilter,blend_chunk,bridge_chunk
from rtc_piper.safety import damped_stop

ROOT=Path(__file__).resolve().parent


class RuntimeTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_rtc_time_axis_interpolates_without_fourfold_speedup(self):
        timeline=Timeline(ROOT,self.cfg,10)
        actions=np.zeros((50,7));actions[:,0]=np.arange(50)*.01
        timeline.enqueue(actions,10,10.08)
        self.assertAlmostEqual(timeline.target(10+1/120)[0],.0025)
        self.assertAlmostEqual(timeline.target(10+1/30)[0],.01)

    def test_lookahead_does_not_expire_current_last_frame(self):
        timeline=Timeline(ROOT,self.cfg,10)
        timeline.enqueue(np.zeros((50,7)),10,10.08)
        horizon=self.cfg['execute_horizon']
        self.assertIsNotNone(timeline.target(10+(horizon-.9)/30))
        self.assertIsNotNone(timeline.target(10+(horizon-.6)/30))
        self.assertIsNone(timeline.target(10+horizon/30+.001))

    def test_expired_results_and_invalid_actions_rejected(self):
        timeline=Timeline(ROOT,self.cfg,10)
        with self.assertRaises(RuntimeError):timeline.enqueue(np.zeros((50,7)),10,10.251)
        for actions in [np.zeros((30,7)),np.full((50,7),np.nan)]:
            with self.assertRaises(RuntimeError):validate_actions(actions,self.cfg)
        actions=np.zeros((50,7));actions[0,6]=-.001
        self.assertEqual(validate_actions(actions,self.cfg)[0,6],0)
        actions[0,6]=-.006
        self.assertEqual(validate_actions(actions,self.cfg)[0,6],0)

    def test_old_predictions_never_survive_rtc_clear(self):
        timeline=Timeline(ROOT,self.cfg,10);generation=timeline.buffer.get_generation()
        timeline.buffer.clear()
        self.assertFalse(timeline.buffer.enqueue(np.zeros((50,7)),0,generation))

    def test_camera_pair_uses_buffered_capture_not_independent_latest(self):
        bank=CaptureBank();rgb=np.zeros((2,2,3),dtype=np.uint8)
        for key,values in [('j12',[1,2]),('j34',[3,4]),('j56',[5,6]),('grip',[.02])]:
            bank.state(key,9.969,values)
        bank.image('cam0',9.967,10,rgb);bank.image('cam0',10,10.03,rgb)
        bank.image('cam1',9.970,10.01,rgb)
        obs=bank.observation(10.04)
        self.assertEqual(obs.t,9.967)
        np.testing.assert_allclose(obs.state,[1,2,3,4,5,6,.02])

    def test_historical_observation_does_not_replace_live_feedback(self):
        bank=CaptureBank();rgb=np.zeros((2,2,3),dtype=np.uint8)
        for key,size in [('j12',2),('j34',2),('j56',2),('grip',1)]:
            bank.state(key,10.04,np.full(size,.01));bank.state(key,10.099,np.full(size,.08))
        bank.image('cam0',10.04,10.08,rgb);bank.image('cam1',10.045,10.08,rgb)
        np.testing.assert_allclose(bank.live(10.1)[0],.08)
        np.testing.assert_allclose(bank.observation(10.1).state,.01)

    def test_missing_old_or_incompatible_sensors_fail_closed(self):
        bank=CaptureBank();rgb=np.zeros((1,1,3),np.uint8)
        with self.assertRaises(NotReady):bank.live(10)
        bank.image('cam0',9.9,9.94,rgb);bank.image('cam1',9.93,9.97,rgb)
        with self.assertRaises(NotReady):bank.observation(10)
        bank.image('cam1',9.9,9.94,rgb)
        with self.assertRaises(NotReady):bank.observation(10)

    def test_nearest_start_is_projected_to_hardware_range_without_dataset_distance_gate(self):
        starts=json.loads((ROOT/'tests/fixtures/start_states.synthetic.json').read_text())
        state=np.r_[np.radians([9.331,-2.052,1.444,3.179,34.547,9.054]),.02]
        row,target=choose_start(state,starts,self.cfg)
        np.testing.assert_allclose(target[:6],np.clip(starts[row][:6],self.cfg['policy_command_lower_rad'],self.cfg['policy_command_upper_rad']))
        self.assertEqual(target[6],.02)
        state[0]=math.radians(-100)
        _,target=choose_start(state,starts,self.cfg)
        self.assertEqual(target[6],.02)

    def test_preparation_speed_acceleration_and_gripper_limits(self):
        start=np.zeros(7);start[6]=.02;goal=start.copy();goal[4]=math.radians(22.221)
        _,duration=trajectory(start,goal,0)
        times=np.linspace(0,duration,10000)
        values=np.array([trajectory(start,goal,t)[0] for t in times])
        velocity=np.diff(np.degrees(values[:,:6]),axis=0)/np.diff(times)[:,None]
        acceleration=np.diff(velocity,axis=0)/np.diff(times)[1:,None]
        self.assertLessEqual(np.max(np.abs(velocity)),5.0001)
        self.assertLessEqual(np.max(np.abs(acceleration)),10.001)
        np.testing.assert_allclose(values[:,6],.02)
        np.testing.assert_allclose(values[-1],goal)

    def test_repeated_feedback_cannot_fake_stationary_duration(self):
        settling=Settling();state=np.zeros(7)
        self.assertFalse(settling.update(state,np.full(4,10)))
        self.assertFalse(settling.update(state,np.full(4,10.01)))
        for _ in range(100):self.assertFalse(settling.update(state,np.full(4,10.01)))
        self.assertTrue(settling.update(state,np.full(4,10.32)))

    def test_policy_rate_limiter_and_gap_protection(self):
        out=limited_target(np.ones(7),np.zeros(7),1/120,self.cfg)
        self.assertAlmostEqual(out[0],math.radians(60)/120)
        self.assertAlmostEqual(out[6],.1/120)
        with self.assertRaises(RuntimeError):limited_target(np.ones(7),np.zeros(7),.101,self.cfg)

    def test_every_sdk_tx_checks_stop_and_rejects_disable(self):
        guard=types.SimpleNamespace(armed=types.SimpleNamespace(value=1),check=mock.Mock())
        bus=mock.Mock();message=types.SimpleNamespace(arbitration_id=0x155,data=bytes(8))
        gated_send(bus,guard,message);bus.send.assert_called_once_with(message,timeout=.02)
        guard.check.side_effect=RuntimeError('stopped')
        with self.assertRaises(RuntimeError):gated_send(bus,guard,message)
        self.assertEqual(bus.send.call_count,1)
        guard.check.side_effect=None
        message=types.SimpleNamespace(arbitration_id=0x471,data=bytes([7,1,0,0,0,0,0,0]))
        with self.assertRaises(RuntimeError):gated_send(bus,guard,message)

    def test_read_only_stage_cannot_send(self):
        guard=types.SimpleNamespace(armed=types.SimpleNamespace(value=0),check=lambda:None)
        with self.assertRaises(RuntimeError):gated_send(mock.Mock(),guard,object())

    def test_stop_path_encodes_only_damping(self):
        packets=[]
        class Bus:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def settimeout(self,value):pass
            def bind(self,channel):self.channel=channel
            def send(self,packet):packets.append(packet);return len(packet)
        with mock.patch.object(socket,'socket',side_effect=lambda *args:Bus()),mock.patch('rtc_piper.safety.time.sleep'):
            result=damped_stop('can1')
        self.assertEqual(result['sent'],10)
        for packet in packets:
            can_id,size,data=struct.unpack('=IB3x8s',packet)
            self.assertEqual((can_id,size,data),(0x150,8,bytes([1,0,0,0,0,0,0,0])))

    def test_preparation_accepts_retained_enable_flags_after_its_own_reset(self):
        from rtc_piper.control import prepare
        from rtc_piper.observations import Observation
        clock=[10.0];events=[];sent_times=[];profiles=[]
        state=np.r_[np.radians([.76,-2.66,2.33,-2.94,32.54,-7.7]),.02]
        physical=dict(estop=True,enabled=True,reset_at=None,enable_at=None)
        class Guard:
            armed=types.SimpleNamespace(value=0)
            def check(self):
                clock[0]+=.0001
                if self.armed.value and clock[0]>self.deadline:raise RuntimeError('Startup deadline')
            def arm(self,startup_seconds=20):
                self.deadline=clock[0]+startup_seconds;self.armed.value=1
        guard=Guard()
        class Bank:
            def live(self):
                clock[0]+=.001
                return state.copy(),np.full(4,clock[0])
            def observation(self):return Observation(clock[0],state.copy(),{},np.full(4,clock[0]))
        class Inner:
            def reset(self):
                events.append('reset');physical['estop']=False  # Flags intentionally remain enabled.
                physical['reset_at']=clock[0]
            def set_motion_mode(self,mode):events.append('mode')
            def enable(self):
                events.append('enable');physical['enabled']=True
                if physical['enable_at'] is None:physical['enable_at']=clock[0]
            def get_arm_status(self):return types.SimpleNamespace(msg=types.SimpleNamespace(ctrl_mode=1,mode_feedback=1))
        class Arm:
            arm=Inner()
            def configure_follower(self):events.append('follower')
            def set_control_profile(self,profile):
                events.append(profile);profiles.append((profile,clock[0],state.copy()))
            def status(self,**kwargs):
                if physical['reset_at'] is not None and clock[0]-physical['reset_at']<.15:
                    raise JointCommunicationFault('joint2 communication',dict(arm_status=5,err_code=2,
                        communication_joints=[2],status_can_hex='0005010000000002'))
                if ('require_enabled' in kwargs and physical['enable_at'] is not None
                        and clock[0]-physical['enable_at']<.18):raise NotReady('Fresh enable confirmation delayed')
                return int(physical['estop']),[physical['enabled']]*6
            def joints(self,values):
                self_test.assertTrue(guard.armed.value)
                state[:6]=values;events.append('joints');sent_times.append(clock[0])
        self_test=self
        starts=json.loads((ROOT/'tests/fixtures/start_states.synthetic.json').read_text())
        with mock.patch('rtc_piper.control.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('rtc_piper.control.time.time',side_effect=lambda:1000+clock[0]), \
                mock.patch('rtc_piper.control.time.sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)), \
                mock.patch('rtc_piper.control._write_recovery'), \
                mock.patch('builtins.print'):
            after,command=prepare(Arm(),types.SimpleNamespace(check=lambda:None),Bank(),guard,starts,self.cfg)
        self.assertEqual(events.count('reset'),1)
        self.assertEqual(events.count('enable'),1)
        self.assertLess(events.index('reset'),events.index('joints'))
        self.assertLess(events.index('follower'),events.index('joints'))
        self.assertLess(events.index('prepare'),events.index('enable'))
        self.assertLess(events.index('enable'),events.index('policy'))
        self.assertLess(events.index('joints'),events.index('enable'))
        enable_holds=np.array([t for t in sent_times if physical['enable_at']<=t<=physical['enable_at']+.18])
        self.assertGreater(len(enable_holds),15)
        self.assertLess(np.max(np.diff(enable_holds)),.02)
        self.assertEqual(command[6],.02)
        np.testing.assert_allclose(command[:6],state[:6],atol=.0001)
        self.assertEqual([p[0] for p in profiles],['prepare','policy'])
        self.assertLess(np.max(np.abs(np.degrees(profiles[-1][2][:6]-command[:6]))),1)

    def test_first_inference_wait_keeps_current_pose_streaming(self):
        from rtc_piper.control import run_policy
        from rtc_piper.observations import Observation
        clock=[10.];commands=[];grips=[];state=np.r_[np.zeros(6),.02]
        def check():
            if clock[0]>=10.22:raise RuntimeError('operator stop')
        guard=types.SimpleNamespace(check=check,stop=types.SimpleNamespace(value=0),
                                    running=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=0))
        bank=types.SimpleNamespace(live=lambda:(state.copy(),np.full(4,clock[0])),
            observation=lambda:Observation(clock[0],state.copy(),{},np.full(4,clock[0])))
        arm=types.SimpleNamespace(status=lambda **kw:None,
            joints=lambda q:commands.append((clock[0],np.array(q))),
            grip=lambda *args:grips.append(clock[0]))
        producer=types.SimpleNamespace(thread=mock.Mock(),error=None,latencies=[],save_report=mock.Mock(),record_feedback=mock.Mock())
        timeline=types.SimpleNamespace(target=lambda now:None if now<10.18 else state.copy())
        settings=self.cfg
        def sleep(seconds):clock[0]+=max(seconds,.0001)
        with mock.patch('rtc_piper.control.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('rtc_piper.control.time.sleep',side_effect=sleep), \
                mock.patch('rtc_piper.control.Timeline',return_value=timeline), \
                mock.patch('rtc_piper.control.Producer',return_value=producer),mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'operator stop'):
                run_policy(ROOT,arm,types.SimpleNamespace(check=lambda:None),bank,guard,
                           object(),state.copy(),9.9,settings)
        hold_times=[t for t,q in commands if t<10.18]
        self.assertGreater(len(hold_times),15)
        self.assertLess(np.max(np.diff(hold_times)),.012)
        for t,q in commands:np.testing.assert_allclose(q,state[:6])
        self.assertTrue(all(t>=10.18 for t in grips))
        reason=producer.save_report.call_args.kwargs['control_error']
        self.assertIn('operator stop',reason['message'])
        self.assertEqual(len(reason['measured_joint_deg']),6)


class CameraRetryTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_temporary_pair_gap_recovers_without_stop_or_fake_policy_heartbeat(self):
        from rtc_piper.observations import Observation
        clock=[10.];calls=[0]
        guard=types.SimpleNamespace(stop=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=9.9))
        def observation():
            calls[0]+=1
            if calls[0]<=2:
                self.assertEqual(guard.policy_time.value,9.9)
                raise NotReady('Waiting for pair',dict(cam1='temporarily absent'))
            return Observation(clock[0],np.zeros(7),{},np.full(4,clock[0]))
        timeline=Timeline(ROOT,self.cfg,10);original=timeline.enqueue
        def enqueue(*args):
            report=original(*args);guard.stop.value=1;return report
        timeline.enqueue=enqueue
        client=types.SimpleNamespace(infer=mock.Mock(return_value=np.zeros((50,7))))
        producer=Producer(client,observation,timeline,guard,self.cfg['prompt'])
        with mock.patch('rtc_piper.policy.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('rtc_piper.policy.time.sleep',side_effect=lambda s:clock.__setitem__(0,clock[0]+s)):
            producer.run()
        self.assertIsNone(producer.error)
        self.assertEqual(producer.stats['observation_waits'],2)
        self.assertEqual(producer.stats['accepted_chunks'],1)
        self.assertTrue(producer.observation_wait['recovered'])
        self.assertAlmostEqual(guard.policy_time.value,clock[0])

    def test_persistent_pair_loss_cannot_extend_action_prefix_or_policy_heartbeat(self):
        clock=[10.08]
        guard=types.SimpleNamespace(stop=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=10.08))
        timeline=Timeline(ROOT,self.cfg,10);timeline.enqueue(np.zeros((50,7)),10,10.08)
        def observation():raise NotReady('Waiting for fresh camera pair')
        def sleep(seconds):
            clock[0]+=seconds
            if timeline.target(clock[0]) is None:guard.stop.value=1
        client=types.SimpleNamespace(infer=mock.Mock())
        producer=Producer(client,observation,timeline,guard,self.cfg['prompt'])
        with mock.patch('rtc_piper.policy.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('rtc_piper.policy.time.sleep',side_effect=sleep):producer.run()
        self.assertIsNone(timeline.target(clock[0]))
        self.assertEqual(guard.policy_time.value,10.08)
        self.assertFalse(producer.observation_wait['recovered'])
        client.infer.assert_not_called()

    def test_camera_driver_failure_still_stops_immediately(self):
        guard=types.SimpleNamespace(stop=types.SimpleNamespace(value=0))
        def observation():raise RuntimeError('Camera failure: device disconnected')
        producer=Producer(object(),observation,object(),guard,self.cfg['prompt']);producer.run()
        self.assertEqual(guard.stop.value,1)
        self.assertIn('disconnected',producer.error)
        self.assertEqual(producer.stats['observation_waits'],0)

    def test_missing_camera_pair_reports_capture_and_arrival_ages(self):
        bank=CaptureBank();bank.image('cam0',9.99,9.995,np.zeros((1,1,3),np.uint8))
        with self.assertRaises(NotReady) as error:bank.observation(10)
        details=error.exception.details
        self.assertEqual(details['images']['cam1']['buffered'],0)
        self.assertAlmostEqual(details['images']['cam0']['capture_age_ms'],10)
        self.assertIsNone(details['min_fresh_pair_skew_ms'])


class HardwareRangeTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_large_finite_forecasts_are_clamped_to_full_firmware_travel(self):
        for value,key in [(1e6,'policy_command_upper_rad'),(-1e6,'policy_command_lower_rad')]:
            raw=np.full((50,7),value);out=validate_actions(raw,self.cfg)
            np.testing.assert_allclose(out[:,:6],np.tile(self.cfg[key],(50,1)))
            self.assertTrue(np.all(out[:,6]==(0.1 if value>0 else 0)))
            self.assertTrue(np.all(raw==value))

    def test_limit_queries_only_send_six_read_requests_and_live_response_is_authoritative(self):
        sent=[];rows=json.loads(json.dumps(self.cfg['hardware_limits_cache']['limits']))
        rows[5]['lower_deg']=-160;rows[5]['upper_deg']=160
        class Bus:
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def settimeout(self,*_):pass
            def setsockopt(self,*_):pass
            def bind(self,*_):pass
            def send(self,packet):
                cid,size,data=struct.unpack('=IB3x8s',packet)
                self.joint=data[0];sent.append((cid,size,data));return len(packet)
            def recv(self,*_):
                row=rows[self.joint-1]
                data=bytes([self.joint])+struct.pack('>hhH',round(row['upper_deg']*10),round(row['lower_deg']*10),300)+bytes(1)
                return struct.pack('=IB3x8s',0x473,8,data)
        with mock.patch.object(socket,'socket',return_value=Bus()):readback=read_hardware_limits(self.cfg)
        self.assertEqual(len(sent),6)
        for joint,(cid,size,data) in enumerate(sent,1):
            self.assertEqual((cid,size,data),(0x472,8,bytes([joint,1,0,0,0,0,0,0])))
        updated=apply_hardware_limits(self.cfg,readback)
        self.assertAlmostEqual(np.degrees(updated['policy_command_lower_rad'][5]),-160)
        self.assertAlmostEqual(np.degrees(updated['policy_command_upper_rad'][5]),160)

    def test_policy_can_run_beyond_ninety_seconds_without_fixed_tracking_angle_stop(self):
        from rtc_piper.control import run_policy
        from rtc_piper.observations import Observation
        clock=[10.];count=[0];measured=np.r_[np.radians([20,0,0,0,0,0]),.02]
        command=np.r_[np.zeros(6),.02]
        def check():
            if clock[0]>=100.1:raise RuntimeError('operator stop')
        def joints(*_):count[0]+=1
        guard=types.SimpleNamespace(check=check,stop=types.SimpleNamespace(value=0),running=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=0))
        bank=types.SimpleNamespace(live=lambda:(measured.copy(),np.full(4,clock[0])),
            observation=lambda:Observation(clock[0],measured.copy(),{},np.full(4,clock[0])))
        arm=types.SimpleNamespace(status=lambda **kw:None,joints=joints,grip=lambda *_:None)
        producer=types.SimpleNamespace(thread=mock.Mock(),error=None,latencies=[],save_report=mock.Mock(),record_feedback=mock.Mock())
        def sleep(seconds):clock[0]+=max(seconds,.0001)
        with mock.patch('rtc_piper.control.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('rtc_piper.control.time.sleep',side_effect=sleep), \
                mock.patch('rtc_piper.control.Timeline',return_value=types.SimpleNamespace(target=lambda _:command.copy())), \
                mock.patch('rtc_piper.control.Producer',return_value=producer),mock.patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'operator stop'):
                run_policy(ROOT,arm,types.SimpleNamespace(check=lambda:None),bank,guard,object(),command.copy(),9.9,self.cfg)
        self.assertGreater(count[0],10000)
        self.assertGreaterEqual(clock[0],100.1)
        self.assertEqual(guard.stop.value,1)


class ResetRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg=config(ROOT/'config.yaml');self.clock=10.;self.trace=dict(events=[])
        self.arm=types.SimpleNamespace(arm=mock.Mock(),status=mock.Mock())
        self.cameras=types.SimpleNamespace(check=mock.Mock())
        self.guard=types.SimpleNamespace(check=mock.Mock())
        self.bank=types.SimpleNamespace(live=lambda:(np.zeros(7),np.full(4,self.clock)))
        self.fault=JointCommunicationFault('joint2 communication',dict(arm_status=5,err_code=2,
            communication_joints=[2],status_can_hex='0005010000000002'))

    def wait(self):
        from rtc_piper.control import wait_after_reset
        def sleep(seconds):self.clock+=seconds
        with mock.patch('rtc_piper.control.time.monotonic',side_effect=lambda:self.clock), \
                mock.patch('rtc_piper.control.time.time',side_effect=lambda:1000+self.clock), \
                mock.patch('rtc_piper.control.time.sleep',side_effect=sleep),mock.patch('builtins.print'):
            return wait_after_reset(self.arm,self.cameras,self.bank,self.guard,1010.,10.,self.cfg,self.trace)

    def test_transient_clears_then_requires_new_stable_feedback(self):
        def status(**kwargs):
            if self.clock<10.15:raise self.fault
            self.assertGreater(kwargs['after'],1010.)
            return 0,[False]*6
        self.arm.status.side_effect=status
        self.wait()
        self.assertGreaterEqual(self.clock,10.45)
        self.assertEqual(self.trace['status'],'passed')
        self.assertEqual(self.trace['events'][0]['communication_joints'],[2])
        self.assertFalse(self.arm.arm.mock_calls)

    def test_persistent_communication_error_never_enables_or_sends_target(self):
        self.arm.status.side_effect=self.fault
        with self.assertRaisesRegex(RuntimeError,'3秒.*关节=\\[2\\]'):self.wait()
        self.assertGreaterEqual(self.clock,13.)
        self.assertEqual(self.trace['status'],'failed')
        self.assertFalse(self.arm.arm.mock_calls)

    def test_other_fault_or_lost_guard_is_not_retried(self):
        for source in (self.arm.status,self.guard.check):
            source.side_effect=RuntimeError('collision or stop')
            with self.assertRaisesRegex(RuntimeError,'collision or stop'):self.wait()
            self.assertEqual(self.clock,10.)
            source.side_effect=None
        self.assertFalse(self.arm.arm.mock_calls)

    def test_recurrence_restarts_entire_stability_window(self):
        def status(**kwargs):
            if 10.2<=self.clock<10.25:raise self.fault
            return 0,[False]*6
        self.arm.status.side_effect=status
        self.wait()
        self.assertGreaterEqual(self.clock,10.55)
        self.assertFalse(self.arm.arm.mock_calls)


class MeasuredStateBoundsTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_recorded_joint5_overshoot_is_a_valid_measured_state(self):
        state=np.r_[np.radians([-13.667,79.524,-3.032,11.627,-36.558,-60.453]),.0728]
        self.assertLess(np.degrees(state[4]),-36.447)
        np.testing.assert_array_equal(check_state(state,self.cfg),state)
        self.assertLess(abs(np.degrees(state[4])-(-36.132)),1)

    def test_small_encoder_overshoot_is_kept_raw_instead_of_becoming_a_second_limit_stop(self):
        for joint,angle in [(3,-127.1),(3,127.1),(4,-89.6),(4,89.6),(5,-170.1),(5,170.1)]:
            with self.subTest(joint=joint,angle=angle):
                state=np.zeros(7);state[joint]=math.radians(angle)
                np.testing.assert_array_equal(check_state(state,self.cfg),state)

    def test_malformed_or_nonfinite_measurements_remain_fatal(self):
        for state in [np.zeros(6),np.full(7,np.nan)]:
            with self.assertRaises(RuntimeError):check_state(state,self.cfg)

    def test_real_minus170_point080_feedback_is_logged_but_target_remains_minus170(self):
        state=np.zeros(7);state[5]=math.radians(-170.080);state[6]=.1001
        check_state(state,self.cfg)
        producer=Producer(object(),object(),object(),object(),self.cfg['prompt'])
        producer.record_feedback(state,self.cfg)
        self.assertAlmostEqual(producer.stats['feedback_max_overshoot_deg'][5],.080)
        raw=np.tile(state,(50,1));bounded=validate_actions(raw,self.cfg)
        self.assertAlmostEqual(np.degrees(bounded[0,5]),-170)
        self.assertEqual(bounded[0,6],.1)
        self.assertAlmostEqual(np.degrees(state[5]),-170.080)


class Kai0TransitionTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_new_chunk_arrival_does_not_jump_current_interpolated_target(self):
        timeline=Timeline(ROOT,self.cfg,10);old=np.zeros((50,7));old[:,0]=np.arange(50)*.001
        timeline.enqueue(old,10,10.08)
        now=10.155;before=timeline.target(now).copy()
        new=old.copy();new[:,0]=.2
        timeline.enqueue(new,10.1,now)
        np.testing.assert_allclose(timeline.target(now),before,atol=1e-12)
        # Start at next30Hz grid boundary5; final transition point is all new.
        last=5+self.cfg['chunk_smoothing_steps']-1
        self.assertAlmostEqual(timeline.target(10+last/30)[0],.2)
        self.assertEqual(len(timeline.buffer.chunks),1)

    def test_initial_hold_is_blended_only_after_a_real_prediction_exists(self):
        hold=np.zeros(7);timeline=Timeline(ROOT,self.cfg,10,initial_action=hold)
        self.assertIsNone(timeline.target(10.08))
        actions=np.zeros((50,7));actions[:,0]=.2
        timeline.enqueue(actions,10,10.08)
        self.assertAlmostEqual(timeline.target(10.08)[0],0)
        self.assertAlmostEqual(timeline.target(10+(3+self.cfg['chunk_smoothing_steps']-1)/30)[0],.2)

    def test_transition_does_not_extend_old_prediction_expiry(self):
        timeline=Timeline(ROOT,self.cfg,10);old=np.zeros((50,7));timeline.enqueue(old,10,10.08)
        new=np.zeros((50,7));new[:,0]=.1
        timeline.enqueue(new,10.1,10.18)
        self.assertIsNone(timeline.target(10.1+self.cfg['execute_horizon']/30+.001))

    def test_short_overlap_reaches_new_plan_before_old_plan_expires(self):
        new=np.ones((50,7))*.2
        result,_=blend_chunk(new,0,2.4,lambda frame:np.zeros(7) if frame<=5 else None,8,18)
        np.testing.assert_allclose(result[3],0)
        np.testing.assert_allclose(result[4],.1)
        np.testing.assert_allclose(result[5:7],.2)

    def test_interpolation_uses_one_snapshot_when_producer_replaces_mid_read(self):
        timeline=Timeline(ROOT,self.cfg,10)
        old=np.zeros((50,7));new=old.copy();new[:,0]=.2
        timeline.enqueue(old,10,10.08)
        get=timeline.peek.get_action;calls=[]
        def interleaved(index):
            value=get(index);calls.append(index)
            if len(calls)==1:
                with timeline.buffer.lock:timeline.buffer.chunks={0:new}
            return value
        with mock.patch.object(timeline.peek,'get_action',side_effect=interleaved):
            self.assertAlmostEqual(timeline.target(10.155)[0],0)
        self.assertAlmostEqual(timeline.target(10.155)[0],.2)


class OutputSmoothingTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_constant_target_is_monotone_without_overshoot_at_irregular_dt(self):
        filt=CriticallyDampedFilter(np.zeros(7),.04);values=[]
        for dt in [.007,.009,.008,.011]*80:values.append(filt.step(np.ones(7),dt))
        values=np.array(values)
        self.assertTrue(np.all(np.diff(values,axis=0)>=-1e-12))
        self.assertTrue(np.all((values>=0)&(values<=1+1e-12)))
        np.testing.assert_allclose(values[-1],1,atol=1e-9)

    def test_ramp_keeps_requested_speed_with_about_eighty_ms_lag(self):
        initial=np.r_[np.radians([0,30,-30,0,0,0]),.02]
        smoother=CommandSmoother(initial,self.cfg);rows=[];dt=1/120
        for t in np.arange(dt,2,dt):
            target=initial.copy();target[0]=math.radians(10*t)
            rows.append((t,smoother.step(target,dt)[0]))
        t,q=np.array(rows).T
        speed=np.degrees(np.diff(q)/np.diff(t))
        self.assertAlmostEqual(np.median(speed[-60:]),10,places=3)
        lag=t[-1]-np.degrees(q[-1])/10
        self.assertGreater(lag,.06);self.assertLess(lag,.1)

    def test_repeated_large_reversals_keep_full_firmware_command_bounds(self):
        low=np.r_[self.cfg['policy_command_lower_rad'],0]
        high=np.r_[self.cfg['policy_command_upper_rad'],.1]
        smoother=CommandSmoother((low+high)/2,self.cfg)
        for i in range(1000):
            q=smoother.step(high+1 if i//10%2 else low-1,1/120)
            self.assertTrue(np.all(q>=low-1e-12));self.assertTrue(np.all(q<=high+1e-12))
        with self.assertRaises(RuntimeError):smoother.step(high,.101)

    def test_filter_reduces_alternating_velocity_changes(self):
        filt=CriticallyDampedFilter(np.zeros(7),.04);raw=[];smooth=[]
        for i in range(600):
            q=np.full(7,.05 if i//10%2 else -.05)
            raw.append(q);smooth.append(filt.step(q,1/120))
        raw_acc=np.diff(np.array(raw)[:,0],n=2)
        smooth_acc=np.diff(np.array(smooth)[:,0],n=2)
        self.assertLess(np.max(abs(smooth_acc)),.1*np.max(abs(raw_acc)))


class VelocityBridgeTests(unittest.TestCase):
    def test_handover_preserves_outgoing_step_instead_of_restarting_at_rest(self):
        old=np.zeros((50,7));old[:,0]=np.arange(50)*.001
        new=old.copy();new[:,0]+=.05
        out,_=bridge_chunk(new,0,4.4,lambda f:old[f] if f<18 else None,10,18)
        np.testing.assert_allclose(out[4:7],old[4:7])
        self.assertAlmostEqual(out[6,0]-out[5,0],.001)
        self.assertAlmostEqual(out[14,0],new[14,0])

    def test_identical_constant_velocity_predictions_do_not_create_slowdowns(self):
        actions=np.zeros((50,7));actions[:,0]=np.arange(50)*.004
        out,_=bridge_chunk(actions,0,4.4,lambda f:actions[f] if f<18 else None,10,18)
        np.testing.assert_allclose(out,actions,atol=1e-12)

    def test_short_overlap_never_pads_old_tail_past_expiry(self):
        old=np.zeros((50,7));old[:,0]=np.arange(50)*.001
        new=old.copy();new[:,0]+=.02
        out,_=bridge_chunk(new,0,4.4,lambda f:old[f] if f<9 else None,10,18)
        np.testing.assert_allclose(out[8:],new[8:])


class FinalCheckpointSelectionTests(unittest.TestCase):
    def test_old_checkpoint_configuration_is_rejected(self):
        settings=json.loads((ROOT/'config.yaml').read_text())
        settings.update(model_step=1500,checkpoint_index=1500,checkpoint='checkpoints/pi05_piper_1500_bf16')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.yaml';path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'2500'):config(path)

    def test_client_refuses_old_service_and_accepts_verified_final_identity(self):
        from rtc_piper.policy import Client
        from openpi_client import msgpack_numpy
        cfg=config(ROOT/'config.yaml')
        metadata=dict(model='pi05_piper_grasp_2500',training_updates=2500,
            source_checkpoint_index=2499,action_horizon=50,physical_action_dim=7,fps=30,warmed_up=True)
        for changed in [dict(model='pi05_piper_grasp_1500'),dict(source_checkpoint_index=2498),{}]:
            ws=mock.Mock();ws.recv.return_value=msgpack_numpy.packb(dict(metadata,**changed))
            with mock.patch('websockets.sync.client.connect',return_value=ws):
                if changed:
                    with self.assertRaisesRegex(RuntimeError,'Wrong policy'):Client(cfg)
                    ws.close.assert_called_once()
                else:
                    client=Client(cfg);self.assertEqual(client.metadata['training_updates'],2500);client.close()


class AutoStartTests(unittest.TestCase):
    def invoke(self,failed=False):
        from rtc_piper.cli import execute
        cfg=config(ROOT/'config.yaml');state=np.r_[np.zeros(6),.02]
        guard=types.SimpleNamespace(start=mock.Mock(),check=mock.Mock(),close=mock.Mock(),
            armed=types.SimpleNamespace(value=0),stop=types.SimpleNamespace(value=0))
        client=mock.Mock();arm=mock.Mock();cameras=mock.Mock();cameras.diagnostics.return_value={}
        prepare=mock.Mock(return_value=(10,state.copy()));run=mock.Mock()
        ready=mock.Mock(return_value=(state,np.full(4,10),1,[False]*6,None))
        if failed:ready.side_effect=RuntimeError('preflight failed')
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            base=Path(folder);(base/'logs').mkdir();(base/'training_start_states.json').write_text(json.dumps([state.tolist()]))
            patches=[('rtc_piper.cli.ROOT',base),('rtc_piper.cli.ensure_policy',mock.Mock(return_value=client)),
                ('rtc_piper.cli.live_settings',mock.Mock(return_value=cfg)),
                ('rtc_piper.cli.signal.signal',mock.Mock(return_value=None)),
                ('rtc_piper.hardware.follower_lock',mock.Mock(side_effect=lambda *_:nullcontext())),
                ('rtc_piper.hardware.check_interface',mock.Mock()),('rtc_piper.hardware.SdkArm',mock.Mock(return_value=arm)),
                ('rtc_piper.observations.RealSense',mock.Mock(return_value=cameras)),
                ('rtc_piper.safety.Guard',mock.Mock(return_value=guard)),
                ('rtc_piper.safety.terminal_mode',mock.Mock(side_effect=lambda *_:nullcontext())),
                ('rtc_piper.control.wait_ready',ready),('rtc_piper.control.prepare',prepare),('rtc_piper.control.run_policy',run),
                ('builtins.input',mock.Mock(side_effect=AssertionError('RUN prompt must not exist'))),('builtins.print',mock.Mock())]
            for target,value in patches:stack.enter_context(mock.patch(target,value))
            stack.enter_context(mock.patch.object(socket,'socket',side_effect=AssertionError('No hardware in auto-entry tests')))
            if failed:
                with self.assertRaisesRegex(RuntimeError,'preflight failed'):execute(cfg)
            else:execute(cfg)
        return prepare,run,guard

    def test_run_starts_after_preflight_without_a_run_prompt(self):
        prepare,run,guard=self.invoke();prepare.assert_called_once();run.assert_called_once();guard.close.assert_called_once()

    def test_failed_preflight_never_prepares_or_runs(self):
        prepare,run,guard=self.invoke(failed=True);prepare.assert_not_called();run.assert_not_called();guard.close.assert_called_once()

    def test_emergency_cancel_aborts_model_connection_before_auto_start(self):
        from rtc_piper.cli import ensure_policy
        guard=types.SimpleNamespace(check=mock.Mock(side_effect=RuntimeError('cancelled')))
        with mock.patch('rtc_piper.policy.Client') as client:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):ensure_policy(config(ROOT/'config.yaml'),guard=guard)
        client.assert_not_called()


class ActionBoundsTests(unittest.TestCase):
    def setUp(self):self.cfg=config(ROOT/'config.yaml')

    def test_observed_small_joint3_overshoot_is_clipped_without_changing_raw_array(self):
        raw=np.zeros((50,7));raw[0:2,2]=math.radians(5.208553)
        original=raw.copy();report={}
        out=validate_actions(raw,self.cfg,report)
        np.testing.assert_array_equal(raw,original)
        np.testing.assert_allclose(np.degrees(out[:2,2]),0)
        self.assertTrue(report['corrected']);self.assertEqual(report['worst']['joint'],3)
        self.assertAlmostEqual(report['worst']['correction_deg'],5.208553)

    def test_wrist_prediction_is_projected_inside_tighter_readback_bounds(self):
        raw=np.zeros((50,7));raw[:,5]=math.radians(-175.8)
        out=validate_actions(raw,self.cfg)
        np.testing.assert_allclose(np.degrees(out[:,5]),-170)
        timeline=Timeline(ROOT,self.cfg,10)
        report=timeline.enqueue(raw,10,10.08)
        self.assertEqual(report['worst']['joint'],6)
        self.assertGreaterEqual(np.degrees(timeline.target(10.1)[5]),-170.000001)

    def test_first_unexecuted_frame_does_not_affect_valid_prefix(self):
        prefix=self.cfg['execute_horizon']
        raw=np.zeros((50,7));raw[prefix,5]=math.radians(-250)
        timeline=Timeline(ROOT,self.cfg,10)
        report=timeline.enqueue(raw,10,10.08)
        self.assertFalse(report['corrected'])
        self.assertAlmostEqual(timeline.target(10+(prefix-.1)/30)[5],0)
        self.assertIsNone(timeline.target(10+prefix/30+.001))

    def test_last_executed_frame_is_clipped_even_for_large_finite_outlier(self):
        prefix=self.cfg['execute_horizon']
        raw=np.zeros((50,7));raw[prefix-1,5]=math.radians(-178)
        timeline=Timeline(ROOT,self.cfg,10)
        timeline.enqueue(raw,10,10.08)
        self.assertAlmostEqual(np.degrees(timeline.target(10+(prefix-.9)/30)[5]),-170)

    def test_fused_targets_and_rate_limited_commands_remain_within_bounds(self):
        low=np.asarray(self.cfg['policy_command_lower_rad']);high=np.asarray(self.cfg['policy_command_upper_rad'])
        raw=np.tile(np.r_[(low+high)/2,.02],(50,1));raw[:,3]=high[3]+math.radians(.5)
        timeline=Timeline(ROOT,self.cfg,10);timeline.enqueue(raw,10,10.08)
        other=raw.copy();other[:,3]=high[3]-math.radians(1)
        timeline.enqueue(other,10.1,10.18)
        previous=(low+high)/2;previous=np.r_[previous,.02]
        for t in np.arange(10.18,10.35,1/120):
            target=timeline.target(float(t));command=limited_target(target,previous,1/120,self.cfg)
            self.assertTrue(np.all(command[:6]>=low));self.assertTrue(np.all(command[:6]<=high))
            self.assertTrue(np.all(np.abs(command[:6]-previous[:6])<=np.radians(self.cfg['policy_joint_speed_deg_s'])/120+1e-10))
            previous=command

    def test_nonfinite_rejection_latches_stop_before_saving_exact_observation_and_raw_actions(self):
        from rtc_piper.observations import Observation
        raw=np.zeros((50,7));raw[9,5]=np.nan
        obs=Observation(10,np.zeros(7),dict(cam0=np.zeros((2,2,3),np.uint8),cam1=np.ones((2,2,3),np.uint8)),np.full(4,10))
        guard=types.SimpleNamespace(stop=types.SimpleNamespace(value=0),policy_time=types.SimpleNamespace(value=0))
        timeline=Timeline(ROOT,self.cfg,10)
        producer=Producer(types.SimpleNamespace(infer=lambda *_:raw),lambda:obs,timeline,guard,self.cfg['prompt'])
        with mock.patch('rtc_piper.policy.time.monotonic',return_value=10.08):producer.run()
        self.assertEqual(guard.stop.value,1)
        self.assertFalse(timeline.buffer.chunks)
        self.assertIn('finite',producer.error)
        with tempfile.TemporaryDirectory() as folder,mock.patch('builtins.print'):
            producer.save_report(folder)
            report=json.loads((Path(folder)/'logs/last_policy_run.json').read_text())
            with np.load(report['rejected_observation'],allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['raw_actions'],raw)
                np.testing.assert_array_equal(saved['state'],obs.state)
                np.testing.assert_array_equal(saved['cam1'],obs.images['cam1'])
            self.assertIn('finite',report['error'])

    def test_malformed_firmware_limit_cache_is_rejected(self):
        settings=json.loads(json.dumps(self.cfg))
        settings['hardware_limits_cache']['limits'][5]['lower_deg']=175
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.yaml';path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError,'Malformed firmware'):config(path)


class VerifiedProgramProtocolTests(unittest.TestCase):
    """Expected bytes come from the working collector and angle monitor."""
    def setUp(self):
        from pyAgxArm import create_agx_arm_config,AgxArmFactory
        self.packets=[]
        settings=config(ROOT/'config.yaml')
        limits={'joint%d'%(i+1):[settings['policy_command_lower_rad'][i],settings['policy_command_upper_rad'][i]] for i in range(6)}
        with mock.patch.object(socket,'socket',side_effect=AssertionError('No CAN sockets in unit tests')):
            self.arm=AgxArmFactory.create_arm(create_agx_arm_config(
                robot='piper_l',firmeware_version='v189',auto_connect=False,
                enable_check_can=False,joint_limits=limits))
        self.arm._ctx.comm=types.SimpleNamespace(send=lambda message:self.packets.append(message),get_channel=lambda:'fake')
        self.arm.set_auto_set_motion_mode_enabled(False)
        self.arm._msg_mode.move_spd_rate_ctrl=100

    def tearDown(self):self.arm._ctx.comm=None

    def test_joint_follow_mode_matches_working_collector(self):
        self.arm.set_motion_mode('js')
        self.assertEqual(self.packets[0].arbitration_id,0x151)
        self.assertEqual(bytes(self.packets[0].data),bytes([1,1,100,0xAD,0,0,0,0]))

    def streaming_bridge(self):
        bridge=SdkArm.__new__(SdkArm);bridge.arm=self.arm
        bridge.guard=types.SimpleNamespace(check=mock.Mock())
        bridge._last_mode=float('-inf');bridge._last_joint_send=None
        bridge._control_profile='policy'
        settings=config(ROOT/'config.yaml')
        bridge._policy_lower=np.asarray(settings['policy_command_lower_rad'])
        bridge._policy_upper=np.asarray(settings['policy_command_upper_rad'])
        self.arm.get_context().set_tx_repeat_min_interval(0x151,0)
        return bridge

    def test_follower_role_matches_working_collector(self):
        bridge=self.streaming_bridge();bridge.configure_follower()
        self.assertEqual([(p.arbitration_id,bytes(p.data)) for p in self.packets],
                         [(0x470,bytes([0xFC,0,0,0,0,0,0,0]))])

    def test_preparation_uses_ordinary_movej_at_five_percent_before_enable(self):
        bridge=self.streaming_bridge();bridge.set_control_profile('prepare')
        with mock.patch('rtc_piper.hardware.time.monotonic',return_value=10):
            bridge.joints(np.zeros(6));self.arm.enable()
        self.assertEqual([p.arbitration_id for p in self.packets],[0x151,0x155,0x156,0x157,0x471])
        self.assertEqual(bytes(self.packets[0].data),bytes([1,1,5,0,0,0,0,0]))
        self.assertFalse(any(p.arbitration_id==0x159 for p in self.packets))

    def test_following_transition_keeps_the_same_hold_target(self):
        bridge=self.streaming_bridge();bridge.set_control_profile('prepare');clock=[10.]
        target=np.radians([2,2,-2,15,32,-19])
        with mock.patch('rtc_piper.hardware.time.monotonic',side_effect=lambda:clock[0]):
            bridge.joints(target);old=list(self.packets);self.packets.clear()
            clock[0]=10.01;bridge.set_control_profile('policy');bridge.joints(target)
        self.assertEqual(bytes(old[0].data),bytes([1,1,5,0,0,0,0,0]))
        self.assertEqual(bytes(self.packets[0].data),bytes([1,1,100,0xAD,0,0,0,0]))
        self.assertEqual([(p.arbitration_id,bytes(p.data)) for p in old[1:]],
                         [(p.arbitration_id,bytes(p.data)) for p in self.packets[1:]])

    def test_stationary_stream_preserves_120hz_targets_and_periodic_mode(self):
        bridge=self.streaming_bridge();clock=[10.];packets=[]
        self.arm._ctx.comm.send=lambda p:packets.append((clock[0],p))
        with mock.patch('rtc_piper.hardware.time.monotonic',side_effect=lambda:clock[0]):
            for index in range(61):
                clock[0]=10+index/120
                bridge.joints(np.zeros(6))
        self.assertEqual(packets[0][1].arbitration_id,0x151)
        mode=[(t,p) for t,p in packets if p.arbitration_id==0x151]
        self.assertGreaterEqual(len(mode),5)
        self.assertLessEqual(len(mode),6)
        self.assertLessEqual(np.max(np.diff([t for t,p in mode])),.109)
        for t,p in mode:self.assertEqual(bytes(p.data),bytes([1,1,100,0xAD,0,0,0,0]))
        self.assertEqual(sum(p.arbitration_id==0x155 for t,p in packets),61)
        self.assertFalse(any(p.arbitration_id==0x159 for t,p in packets))

    def test_stalled_stream_or_latched_stop_cannot_resume_targets(self):
        bridge=self.streaming_bridge();clock=[10.]
        with mock.patch('rtc_piper.hardware.time.monotonic',side_effect=lambda:clock[0]):
            bridge.joints(np.zeros(6));before=len(self.packets)
            clock[0]=10.101
            with self.assertRaisesRegex(RuntimeError,'stream gap'):bridge.joints(np.zeros(6))
            self.assertEqual(len(self.packets),before)
            bridge.guard.check.side_effect=RuntimeError('latched stop')
            clock[0]=10.02
            with self.assertRaisesRegex(RuntimeError,'latched stop'):bridge.joints(np.zeros(6))
            self.assertEqual(len(self.packets),before)

    def test_final_sdk_send_rejects_wrist_commands_outside_tighter_bounds(self):
        bridge=self.streaming_bridge()
        for joint,value in [(3,127.1),(4,89.6),(5,-170.1)]:
            command=np.zeros(6);command[joint]=math.radians(value)
            with self.assertRaisesRegex(RuntimeError,'live firmware limits'):bridge.joints(command)
            self.assertFalse(self.packets)

    def test_full_hardware_range_is_available_in_preparation_and_policy(self):
        bridge=self.streaming_bridge();bridge.set_control_profile('prepare')
        state=np.zeros(6);state[4]=math.radians(-36.558)
        with mock.patch('rtc_piper.hardware.time.monotonic',return_value=10):bridge.joints(state)
        packet=next(p for p in self.packets if p.arbitration_id==0x157)
        self.assertEqual(struct.unpack('>ii',bytes(packet.data))[0],-36558)
        count=len(self.packets);bridge.set_control_profile('policy')
        with mock.patch('rtc_piper.hardware.time.monotonic',return_value=10.01):bridge.joints(state)
        self.assertGreater(len(self.packets),count)

    def test_joint_targets_match_radians_to_millidegrees_contract(self):
        joints=np.radians([-9.331,2.052,-1.444,-3.179,34.547,-9.054])
        self.arm.move_js(joints.tolist())
        raw=np.rint(joints*180000/math.pi).astype(int)
        self.assertEqual([p.arbitration_id for p in self.packets],[0x155,0x156,0x157])
        for index,packet in enumerate(self.packets):
            self.assertEqual(bytes(packet.data),struct.pack('>ii',*raw[index*2:index*2+2]))

    def test_enable_and_reset_match_working_startup(self):
        self.arm.enable();self.arm.reset()
        self.assertEqual([(p.arbitration_id,bytes(p.data)) for p in self.packets],
                         [(0x471,bytes([7,2,0,0,0,0,0,0])),(0x150,bytes([2,0,0,0,0,0,0,0]))])

    def test_gripper_is_metres_and_not_scaled_twice(self):
        grip=self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        grip.move_gripper_m(value=.04,force=.5)
        packet=self.packets[-1]
        self.assertEqual(packet.arbitration_id,0x159)
        self.assertEqual(bytes(packet.data),struct.pack('>iHBB',40000,500,1,0))

    def test_feedback_matches_angle_monitor_millidegree_decode(self):
        import can
        expected=[9331,-2052,1444,3179,34547,9054]
        for pair in range(3):
            self.arm._parser.parse_packet(can.Message(arbitration_id=0x2A5+pair,
                data=struct.pack('>ii',*expected[pair*2:pair*2+2]),timestamp=100,is_extended_id=False))
        angles=self.arm.get_joint_angles().msg
        np.testing.assert_allclose(np.degrees(angles),np.array(expected)/1000,atol=1e-8)

    def bridge_with_status(self,hex_status):
        import can
        for can_id,data in [(0x2A1,bytes.fromhex(hex_status))]+[
                (0x261+i,bytes.fromhex('00e600231d000000')) for i in range(6)]:
            self.arm._parser.parse_packet(can.Message(arbitration_id=can_id,data=data,
                timestamp=100,is_extended_id=False))
        bridge=SdkArm.__new__(SdkArm);bridge.arm=self.arm;bridge.errors=[]
        bridge.gripper=types.SimpleNamespace(get_gripper_status=lambda:types.SimpleNamespace(
            timestamp=100,msg=types.SimpleNamespace(foc_status=types.SimpleNamespace())))
        return bridge

    def test_screenshot_joint2_fault_still_fails_ordinary_runtime_checks(self):
        bridge=self.bridge_with_status('0005010000000002')
        with mock.patch.object(self.arm,'has_comm_error',return_value=False), \
                mock.patch('rtc_piper.hardware.time.time',return_value=100):
            for kwargs in ({},{'allow_estop':True},{'require_enabled':True}):
                with self.assertRaises(JointCommunicationFault) as context:bridge.status(**kwargs)
                self.assertEqual(context.exception.details['communication_joints'],[2])
                self.assertEqual(context.exception.details['status_can_hex'],'0005010000000002')

    def test_angle_limit_or_driver_fault_is_never_a_reset_communication_wait(self):
        bridge=self.bridge_with_status('0005010000000102')
        with mock.patch.object(self.arm,'has_comm_error',return_value=False), \
                mock.patch('rtc_piper.hardware.time.time',return_value=100):
            with self.assertRaises(RuntimeError) as context:bridge.status(allow_estop=True)
            self.assertNotIsInstance(context.exception,JointCommunicationFault)
        bridge=self.bridge_with_status('0005010000000002')
        import can
        self.arm._parser.parse_packet(can.Message(arbitration_id=0x262,
            data=bytes.fromhex('00e600231d010000'),timestamp=100,is_extended_id=False))
        with mock.patch.object(self.arm,'has_comm_error',return_value=False), \
                mock.patch('rtc_piper.hardware.time.time',return_value=100):
            with self.assertRaisesRegex(RuntimeError,'driver fault') as context:bridge.status(allow_estop=True)
            self.assertNotIsInstance(context.exception,JointCommunicationFault)


if __name__=='__main__':unittest.main()
