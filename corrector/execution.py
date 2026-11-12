"""VLA-Corrector integration with existing RTC, smoothing and damping guard."""
from collections import deque
import json
from pathlib import Path
import threading
import time
import numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.policy import Timeline,CommandSmoother
from rtc_piper.control import check_state


class ExecutionState:
    def __init__(self,root,settings,initial,epoch):
        self.root,self.settings,self.epoch=root,settings,epoch
        self.command=np.array(initial,dtype=float);self.commands=deque([(epoch-1,self.command.copy())],maxlen=2400)
        self.timeline=None;self.generation=0;self.capture_after=epoch-1e-6
        self.last_event_id=0;self.lock=threading.Lock();self.recovery_started=None
        self.events=deque(maxlen=512);self.stats=dict(accepted=0,discarded_stale=0,recoveries=0,monitor_calls=0)

    def command_at(self,stamp):
        with self.lock:
            if stamp<self.commands[0][0]:raise NotReady('No executed command aligned to image capture')
            for t,a in reversed(self.commands):
                if t<=stamp:return a.copy()

    def record_command(self,stamp,command):
        with self.lock:self.command=np.array(command,copy=True);self.commands.append((stamp,self.command.copy()))

    def apply_event(self,result):
        with self.lock:
            if result['event_id']>self.last_event_id:
                self.last_event_id=result['event_id'];self.events.append(dict(t=time.monotonic(),event_id=result['event_id'],
                    generation=result['generation'],observed_at=result.get('event_at'),monitor=result.get('monitor')))
            if result['generation']>self.generation:
                self.generation=int(result['generation']);self.capture_after=float(result['event_at'])
                self.timeline=None;self.recovery_started=time.monotonic();self.stats['recoveries']+=1

    def publish(self,result,captured,sent_generation,now):
        self.apply_event(result)
        with self.lock:
            if result.get('needs_replan') or sent_generation!=self.generation or captured<=self.capture_after:
                self.stats['discarded_stale']+=1;return False
            if self.timeline is None:self.timeline=Timeline(self.root,self.settings,self.epoch,initial_action=self.command)
            self.timeline.enqueue(result['actions'],captured,now)
            self.stats['accepted']+=1;self.recovery_started=None
            return True

    def sample(self,now):
        with self.lock:
            target=None if self.timeline is None else self.timeline.target(now)
            return target,self.recovery_started


def run(root,arm,cameras,bank,guard,policy,observer,previous,after,settings):
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True);arm.joints(previous[:6])
        try:first=bank.observation()
        except NotReady:time.sleep(1/120);continue
        if first.t>after and np.all(first.state_times>after):break
        time.sleep(1/120)
    state=ExecutionState(root,settings,previous,first.t);smoother=CommandSmoother(previous,settings)
    errors=[];stop=threading.Event();rtts=deque(maxlen=1000);monitor_rtts=deque(maxlen=1000)
    def inference_worker():
        last=float('-inf')
        try:
            while not stop.is_set() and not guard.stop.value:
                try:
                    obs=bank.observation()
                    with state.lock:generation=state.generation;cutoff=state.capture_after
                    if obs.t<=max(last,cutoff):raise NotReady('Waiting for a fresh recovery observation')
                    command=state.command_at(obs.t)
                except NotReady:time.sleep(.005);continue
                last=obs.t;before=time.monotonic()
                result=policy.request('infer',obs,command,generation,settings['prompt']);now=time.monotonic()
                if stop.is_set() or guard.stop.value:return
                if state.publish(result,obs.t,generation,now):guard.policy_time.value=now
                rtts.append(now-before)
        except BaseException as error:
            if not stop.is_set():errors.append(repr(error));guard.stop.value=1
    def monitor_worker():
        last=float('-inf');next_call=time.monotonic()
        try:
            while not stop.is_set() and not guard.stop.value:
                now=time.monotonic()
                if now<next_call:time.sleep(min(.005,next_call-now));continue
                try:
                    obs=bank.observation()
                    if obs.t<=last:raise NotReady('No new monitor frame')
                    command=state.command_at(obs.t)
                except NotReady:time.sleep(.005);continue
                last=obs.t;before=time.monotonic()
                result=observer.request('observe',obs,command,state.generation,settings['prompt'])
                if stop.is_set() or guard.stop.value:return
                state.apply_event(result);state.stats['monitor_calls']+=1
                monitor_rtts.append(time.monotonic()-before);next_call=before+.1
        except BaseException as error:
            if not stop.is_set():errors.append(repr(error));guard.stop.value=1
    threads=[threading.Thread(target=inference_worker,daemon=True),threading.Thread(target=monitor_worker,daemon=True)]
    for thread in threads:thread.start()
    started=None;previous_time=time.monotonic();next_report=0;max_gap=0.;control_error=None
    guard.startup_deadline.value=time.monotonic()+3
    print('π0.5 + Corrector：mode=%s；Space/Esc/Ctrl+C 阻尼停止。'%policy.session.mode,flush=True)
    try:
        while True:
            if errors:raise RuntimeError('Corrector worker failed: '+errors[0])
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();measured,stamps=bank.live();check_state(measured,settings)
            target,recovery_started=state.sample(now)
            if target is None:
                if started is not None and recovery_started is None:raise RuntimeError('Corrector RTC action prefix exhausted')
                if recovery_started is not None and now-recovery_started>.8:raise RuntimeError('Corrector recovery timed out')
                target=previous
            else:
                if started is None:started=now;guard.running.value=1;guard.policy_time.value=now
            dt=max(now-previous_time,1e-6);max_gap=max(max_gap,dt)
            command=smoother.step(target,dt);guard.check()
            arm.joints(command[:6])
            if started is not None:arm.grip(command[6],settings['gripper_force_n'])
            state.record_command(time.monotonic(),command);previous=command;previous_time=now
            if now>=next_report:
                print('Corrector mode=%s chunks=%d events=%d recoveries=%d RTT=%.0fms'%(policy.session.mode,
                    state.stats['accepted'],state.last_event_id,state.stats['recoveries'],1000*np.median(rtts) if rtts else 0),flush=True)
                next_report=now+2
            time.sleep(max(0,1/120-(time.monotonic()-now)))
    except BaseException as error:guard.stop.value=1;control_error=repr(error);raise
    finally:
        guard.stop.value=1;stop.set();policy.close();observer.close()
        for thread in threads:thread.join(.3)
        logs=Path(root)/'logs';logs.mkdir(exist_ok=True)
        record=dict(mode=policy.session.mode,policy_metadata=policy.metadata,stats=state.stats,events=list(state.events),
            error=control_error,worker_errors=errors,max_control_gap_ms=1000*max_gap,
            policy_rtt_ms=[1000*x for x in rtts],monitor_rtt_ms=[1000*x for x in monitor_rtts])
        (logs/'last_corrector_run.json').write_text(json.dumps(record,indent=2)+'\n')
