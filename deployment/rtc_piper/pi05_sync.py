"""Pi05 RTC ablation: settle, fresh observation, infer, execute all50 frames."""
import json
import math
from pathlib import Path
import time
import numpy as np
from rtc_piper.control import check_state
from rtc_piper.observations import NotReady
from rtc_piper.policy import CommandSmoother,validate_actions
from rtc_piper.smoothing import bridge_chunk
from rtc_piper.vpp2_control import Steady,InferenceJob


class SyncChunks:
    def __init__(self,client,initial,after,settings,now,job_factory=InferenceJob):
        self.client=client;self.hold=np.array(initial,copy=True);self.settings=settings
        self.after=after;self.stage='settle';self.stable=Steady();self.job=None
        self.deadline=now+settings['settle_timeout_s'];self.job_factory=job_factory
        self.requests=0;self.accepted=0;self.completed=0;self.discarded=0
        self.actions=None;self.play_started=None;self.last_inference=None

    def target(self,now,state,stamps,observation):
        target=self.hold
        if self.stage=='settle':
            if self.stable.update(state,stamps):self.stage='observe';self.after=now
            elif now>self.deadline:raise RuntimeError('pi05 synchronous settling timed out')
        elif self.stage=='observe':
            if now>self.deadline:raise RuntimeError('pi05 synchronous fresh observation timed out')
            if not self.stable.update(state,stamps):self.stage='settle';self.stable=Steady()
            elif (observation is not None and observation.t>self.after
                    and np.all(observation.state_times>self.after)):
                self.job=self.job_factory(self.client,observation,self.settings['prompt'])
                self.requests+=1;self.deadline=now+self.settings['request_timeout_s'];self.stage='infer'
        elif self.stage=='infer':
            if now>self.deadline:raise RuntimeError('pi05 synchronous inference timed out')
            if self.job.done.is_set():
                if self.job.error:raise RuntimeError('pi05 synchronous inference failed: '+str(self.job.error))
                source=self.job.observation
                if (np.max(abs(state[:6]-source.state[:6]))>math.radians(1)
                        or abs(state[6]-source.state[6])>.005):
                    self.discarded+=1;self.stage='settle';self.stable=Steady()
                    self.deadline=now+self.settings['settle_timeout_s']
                else:
                    self.actions=validate_actions(self.job.actions,self.settings)
                    if self.settings.get('smoothing_enabled',True):
                        self.actions,_=bridge_chunk(self.actions,0,0,lambda _:self.hold,
                            self.settings['chunk_smoothing_steps'],50)
                        self.actions=validate_actions(self.actions,self.settings)
                    self.accepted+=1;self.last_inference=now;self.play_started=now
                    self.stage='play';target=self.actions[0]
        elif self.stage=='play':
            frame=(now-self.play_started)*self.settings['fps']
            if frame>=50:
                self.hold=self.actions[-1].copy();target=self.hold;self.completed+=1
                self.stage='settle';self.stable=Steady();self.after=now
                self.deadline=now+self.settings['settle_timeout_s']
            else:
                index=math.floor(frame);right=min(index+1,49)
                target=self.actions[index]+(self.actions[right]-self.actions[index])*(frame-index)
        return np.array(target,copy=True)


def run_sync(root,arm,cameras,bank,guard,client,previous,after,settings):
    now=time.monotonic();plan=SyncChunks(client,previous,after,settings,now)
    smoother=CommandSmoother(previous,settings);previous_time=now;next_notice=0;error=None
    guard.startup_deadline.value=now+settings['settle_timeout_s']+settings['request_timeout_s']+2
    print('π0.5同步：完整50帧→停稳→新观测→推理；不预取、不RTC重叠；Space/Esc/Ctrl+C阻尼停止。',flush=True)
    try:
        while True:
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();state,stamps=bank.live();check_state(state,settings)
            try:obs=bank.observation()
            except NotReady:obs=None
            accepted=plan.accepted;target=plan.target(now,state,stamps,obs)
            if plan.accepted>accepted:
                guard.policy_time.value=now;guard.running.value=1
            command=smoother.step(target,max(now-previous_time,1e-6))
            arm.joints(command[:6]);arm.grip(command[6],settings['gripper_force_n'])
            previous_time=now
            if now>=next_notice:
                print('π0.5同步 phase=%s chunks=%d，平滑%s。'%(plan.stage,plan.completed,
                    '开启' if settings.get('smoothing_enabled',True) else '关闭'),flush=True);next_notice=now+2
            time.sleep(max(0,1/settings['control_hz']-(time.monotonic()-now)))
    except BaseException as exc:guard.stop.value=1;error=str(exc);raise
    finally:
        guard.stop.value=1
        if plan.job:plan.job.thread.join(.1)
        record=dict(time=time.time(),execution_mode='sync_chunks',model_step=settings['model_step'],
            smoothing_enabled=settings.get('smoothing_enabled',True),policy_metadata=client.metadata,
            requests=plan.requests,accepted=plan.accepted,completed=plan.completed,
            discarded_moved_plans=plan.discarded,phase=plan.stage,error=error)
        logs=Path(root)/'logs';logs.mkdir(exist_ok=True)
        (logs/'last_policy_run.json').write_text(json.dumps(record,indent=2)+'\n')
