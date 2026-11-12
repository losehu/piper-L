"""Synchronous VPP2 chunks: finish, stop, observe, infer, then execute.

The14B WAM is slower than its32-frame prediction window on this GPU. Waiting
for a result therefore holds the arm; old observations are never relabelled as
fresh RTC inputs. A worker keeps the hold/keyboard loop responsive, but there
is no inference prefetch during chunk execution.
"""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import numpy as np

from rtc_piper.control import Settling,check_state
from rtc_piper.observations import NotReady
from rtc_piper.policy import validate_actions,CommandSmoother
from rtc_piper.smoothing import bridge_chunk


class InferenceJob:
    def __init__(self,client,observation,prompt):
        self.observation=observation;self.actions=None;self.error=None
        self.done=threading.Event()
        def work():
            try:self.actions=client.infer(observation,prompt)
            except BaseException as error:self.error=error
            finally:self.done.set()
        self.thread=threading.Thread(target=work,daemon=True)
        self.thread.start()


class Steady:
    def __init__(self):self.joints=Settling();self.grip=None;self.grip_speed=float('inf')

    def update(self,state,stamps):
        if self.grip is not None:
            value,stamp=self.grip;dt=stamps[3]-stamp
            if dt>0:self.grip_speed=abs(state[6]-value)/dt
        if self.grip is None or stamps[3]>self.grip[1]:self.grip=(state[6],stamps[3])
        return self.joints.update(state,stamps,position_ok=self.grip_speed<=.002)


def run_vpp2(root,arm,cameras,bank,guard,client,previous,after,settings):
    if settings.get('execution_mode')=='h100_async_rtc':
        from rtc_piper.vpp2_rtc import run_remote_rtc
        return run_remote_rtc(root,arm,cameras,bank,guard,client,previous,after,settings)
    root=Path(root);smoother=CommandSmoother(previous,settings)
    now=time.monotonic();previous_time=now;stage='settle';stable=Steady()
    deadline=now+settings['settle_timeout_s'];hold=previous.copy()
    guard.startup_deadline.value=now+settings['settle_timeout_s']+settings['request_timeout_s']+2
    job=None;actions=None;execution_started=None;capture_after=after
    stats=dict(requests=0,accepted_chunks=0,completed_chunks=0,discarded_moved_plans=0,projected_chunks=0)
    events=deque(maxlen=256);history=deque(maxlen=1200);latencies=[]
    control_error=None;state=previous.copy();next_notice=0
    print('VPP2同步模式：执行完整32帧→停稳保持→采集新观测→推理；控制/急停持续120Hz。',flush=True)
    try:
        while True:
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();state,stamps=bank.live();check_state(state,settings)
            try:observation=bank.observation()
            except NotReady:observation=None
            target=hold
            if stage=='settle':
                settled=stable.update(state,stamps)
                if settled:
                    # Images selected before this point can precede physical
                    # stopping. Observe only after the held arm is stable.
                    capture_after=now;stage='observe'
                    events.append(dict(t=now,stage='stopped'))
                elif now>deadline:raise RuntimeError('VPP2 waiting for stable arm timed out')
            elif stage=='observe':
                if now>deadline:raise RuntimeError('VPP2 waiting for a fresh stopped observation timed out')
                if not stable.update(state,stamps):
                    stage='settle';stable=Steady()
                elif (observation is not None and observation.t>capture_after
                        and np.all(observation.state_times>capture_after)):
                    job=InferenceJob(client,observation,settings['prompt'])
                    infer_started=now;deadline=now+settings['request_timeout_s'];stage='infer'
                    stats['requests']+=1
                    events.append(dict(t=now,stage='infer',observed_at=observation.t))
            elif stage=='infer':
                if now>deadline:raise RuntimeError('VPP2 inference timed out while holding position')
                if job.done.is_set():
                    if job.error:raise RuntimeError('VPP2 inference failed: '+str(job.error)) from job.error
                    if observation is not None:
                        source=job.observation
                        # A planning observation is valid only for the held pose.
                        # If externally moved, discard and observe again, not a
                        # workspace restriction or a silent action/state rebase.
                        moved=(np.max(abs(state[:6]-source.state[:6]))>math.radians(1)
                               or abs(state[6]-source.state[6])>.005)
                        if moved:
                            stats['discarded_moved_plans']+=1
                            stage='settle';stable=Steady();capture_after=now
                            deadline=now+settings['settle_timeout_s']
                            print('VPP2等待期间姿态改变，丢弃旧预测并重新观测。',flush=True)
                        else:
                            report={};actions=validate_actions(job.actions,settings,report)
                            actions,_=bridge_chunk(actions,0,0,lambda _:previous,
                                settings['chunk_smoothing_steps'],32)
                            actions=validate_actions(actions,settings)
                            stats['accepted_chunks']+=1
                            if report['corrected']:stats['projected_chunks']+=1
                            latencies.append(now-infer_started)
                            history.append((source.t,now,np.array(job.actions,dtype=np.float32)))
                            guard.policy_time.value=now;guard.running.value=1
                            execution_started=now;stage='play';target=actions[0]
                            events.append(dict(t=now,stage='play',observed_at=source.t,
                                               inference_s=now-infer_started))
            elif stage=='play':
                frame=(now-execution_started)*30
                if frame>=32:
                    hold=actions[-1].copy();target=hold
                    stats['completed_chunks']+=1
                    stage='settle';stable=Steady();capture_after=now
                    deadline=now+settings['settle_timeout_s']
                    events.append(dict(t=now,stage='chunk_complete'))
                else:
                    index=math.floor(frame);right=min(index+1,31)
                    target=actions[index]+(actions[right]-actions[index])*(frame-index)
            command=smoother.step(target,max(now-previous_time,1e-6))
            arm.joints(command[:6]);arm.grip(command[6],settings['gripper_force_n'])
            previous=command;previous_time=now
            if now>=next_notice:
                rtt=1000*np.median(latencies[-10:]) if latencies else 0
                print('VPP2 phase=%s chunks=%d RTT=%.0fms；Space/Esc/Ctrl+C 阻尼停止。'%(
                    stage,stats['accepted_chunks'],rtt),flush=True)
                next_notice=now+2
            time.sleep(max(0,1/120-(time.monotonic()-now)))
    except BaseException as error:
        guard.stop.value=1
        control_error=dict(message=str(error),type=type(error).__name__,phase=stage,
            measured_joint_deg=np.degrees(state[:6]).tolist(),gripper_m=float(state[6]),
            last_joint_command_deg=np.degrees(previous[:6]).tolist(),last_gripper_command_m=float(previous[6]))
        raise
    finally:
        guard.stop.value=1
        if job:job.thread.join(.1)
        try:
            logs=root/'logs';logs.mkdir(exist_ok=True);stamp=time.strftime('%Y%m%d_%H%M%S')
            record=dict(time=time.time(),model_name='vpp2',policy_metadata=client.metadata,stats=stats,
                control_error=control_error,execution_mode='plan_execute',recent_events=list(events),
                request_rtt_ms=[1000*x for x in latencies],
                execution_settings={key:settings[key] for key in ['model_step','action_horizon','fps',
                    'control_hz','chunk_smoothing_steps','command_filter_tau_s','request_timeout_s','policy_timeout_s',
                    'denoising_steps','compile_vpp2']})
            if history:
                path=logs/('vpp2_policy_chunks_'+stamp+'.npz');rows=list(history)
                np.savez_compressed(path,capture_times=np.array([r[0] for r in rows]),
                    execution_start_times=np.array([r[1] for r in rows]),raw_actions=np.array([r[2] for r in rows]))
                record['prediction_history']=str(path)
            content=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
            (logs/'last_policy_run.json').write_text(content)
            (logs/('policy_run_vpp2_'+stamp+'.json')).write_text(content)
        except (OSError,ValueError) as error:print('VPP2诊断保存失败：',error,flush=True)
