"""VPP2-specific asynchronous, capture-aligned action windows.

The first forecast is planned while the arm is held, then scheduled explicitly
at motion start. Later forecasts keep the real observation time: already-past
samples are skipped, and validity ends at capture + 32/30 seconds. Cubic
position/tangent corrections join valid new forecasts without extending old
ones. The pi05 time axis and its parameters are not modified.
"""
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.policy import validate_actions,CommandSmoother
from rtc_piper.control import check_state


class Window:
    def __init__(self,actions,start,captured,received,settings,anchor,velocity):
        self.actions=np.asarray(actions,dtype=float).copy()
        self.start=float(start);self.captured=float(captured);self.received=float(received)
        self.end=self.start+len(self.actions)/settings['fps'];self.settings=settings
        remaining=self.end-self.received
        if remaining<=0:raise RuntimeError('VPP2 response has no unexpired actions')
        self.duration=min(settings['chunk_smoothing_steps']/settings['fps'],max(0,remaining-1/settings['fps']))
        value,slope=self.base(self.received)
        self.offset=np.asarray(anchor)-value;self.velocity_offset=np.asarray(velocity)-slope

    def base(self,now):
        frame=(now-self.start)*self.settings['fps']
        index=min(max(math.floor(frame),0),len(self.actions)-1);right=min(index+1,len(self.actions)-1)
        slope=(self.actions[right]-self.actions[index])*self.settings['fps']
        return self.actions[index]+(self.actions[right]-self.actions[index])*max(0,frame-index),slope

    def evaluate(self,now):
        if not self.start<=now<self.end:return None
        value,velocity=self.base(now)
        if self.duration>0 and self.received<=now<self.received+self.duration:
            s=(now-self.received)/self.duration
            value=value+(2*s**3-3*s**2+1)*self.offset+(s**3-2*s**2+s)*self.duration*self.velocity_offset
            velocity=velocity+(6*s*s-6*s)/self.duration*self.offset+(3*s*s-4*s+1)*self.velocity_offset
        lower=np.r_[self.settings['policy_command_lower_rad'],0]
        upper=np.r_[self.settings['policy_command_upper_rad'],.1]
        bounded=np.clip(value,lower,upper)
        velocity=np.where(abs(bounded-value)>1e-10,0,velocity)
        return bounded,velocity


class CaptureTimeline:
    def __init__(self,initial,settings):
        self.initial=np.asarray(initial,dtype=float).copy();self.settings=settings
        self.window=None;self.lock=threading.Lock();self.last_capture=float('-inf')
        self.accepted=0;self.last_report=None

    def publish(self,actions,captured,received):
        report={};actions=validate_actions(actions,self.settings,report)
        if not math.isfinite(captured+received) or not 0<=received-captured<self.settings['action_horizon']/self.settings['fps']:
            raise RuntimeError('VPP2 response outside its prediction window')
        with self.lock:
            if captured<=self.last_capture:raise RuntimeError('VPP2 response observation did not advance')
            bootstrap=self.window is None
            prior=None if bootstrap else self.window.evaluate(received)
            if not bootstrap and prior is None:raise RuntimeError('VPP2 RTC action window exhausted before replacement')
            anchor,velocity=(self.initial,np.zeros(7)) if bootstrap else prior
            speeds=np.r_[np.radians(self.settings['policy_joint_speed_deg_s']),self.settings['gripper_speed_m_s']]
            velocity=np.clip(velocity,-speeds,speeds)
            # Holding before the first forecast is explicit scheduling, not a
            # rewritten camera stamp. Every moving update uses captured itself.
            start=received if bootstrap else captured
            window=Window(actions,start,captured,received,self.settings,anchor,velocity)
            report.update(bootstrap=bootstrap,captured_at=captured,received_at=received,
                scheduled_start=start,valid_until=window.end,age_s=received-captured,
                skipped_prefix_frames=0 if bootstrap else math.floor((received-captured)*self.settings['fps']),
                overlap_remaining_s=None if bootstrap else self.window.end-received,
                bridge_s=window.duration)
            self.window=window;self.last_capture=captured;self.accepted+=1;self.last_report=report
        return report

    def sample(self,now):
        with self.lock:window=self.window
        if window is None:return None
        result=window.evaluate(now)
        return None if result is None else result[0]


class Producer:
    def __init__(self,client,observation,live_state,timeline,guard,settings):
        self.client,self.observation,self.live_state=client,observation,live_state
        self.timeline,self.guard,self.settings=timeline,guard,settings
        self.error=None;self.events=deque(maxlen=512);self.latencies=[]
        self.stats=dict(requests=0,accepted_chunks=0,observation_waits=0,bootstrap_discards=0,
            inflight=0,peak_inflight=0,discarded_out_of_order=0,startup_warmup_calls=0)
        self.lock=threading.Lock();self.bootstrap=threading.Event();self.bootstrap_received=float('-inf')
        self.first_moving=threading.Event();self.second_worker_after=float('inf');self.roundtrips=deque(maxlen=8)
        self.next_moving_send=float('-inf')
        self.last_sent=float('-inf')
        self.warmups=int(settings.get('vpp2_startup_warmups',0))
        self.thread=threading.Thread(target=self.run,daemon=True)

    def run(self):
        count=int(self.settings.get('vpp2_max_inflight',1))
        if count==1:return self.worker(0)
        threads=[threading.Thread(target=self.worker,args=(i,),daemon=True) for i in range(count)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()

    def worker(self,index):
        try:
            while not self.guard.stop.value:
                # Offset the two request loops by half a measured round trip.
                # Starting them together creates pairs of almost-identical
                # capture times followed by a full RTT without a new forecast.
                if index>0:
                    if not self.first_moving.wait(.01):continue
                    if time.monotonic()<self.second_worker_after:time.sleep(.005);continue
                obs=None
                with self.lock:
                    try:
                        if (self.bootstrap.is_set() and int(self.settings.get('vpp2_max_inflight',1))>1
                                and time.monotonic()<self.next_moving_send):raise NotReady('Pacing the two remote connections')
                        candidate=self.observation()
                    except NotReady:self.stats['observation_waits']+=1
                    else:
                        if candidate.t>self.last_sent and candidate.t>=self.bootstrap_received-1e-8:
                            obs=candidate;self.last_sent=obs.t;self.stats['requests']+=1
                            self.stats['inflight']+=1
                            self.stats['peak_inflight']=max(self.stats['peak_inflight'],self.stats['inflight'])
                            if index==0 and self.bootstrap.is_set() and not self.first_moving.is_set():
                                delay=float(np.median(self.roundtrips))/2 if self.roundtrips else 0
                                self.second_worker_after=time.monotonic()+delay
                                self.stats['startup_pipeline_offset_s']=delay;self.first_moving.set()
                            if self.bootstrap.is_set() and self.roundtrips:
                                self.next_moving_send=time.monotonic()+float(np.median(self.roundtrips))/2
                if obs is None:time.sleep(.005);continue
                before=time.monotonic()
                try:
                    actions=self.client.infer(obs,self.settings['prompt']);now=time.monotonic()
                    if self.guard.stop.value:return
                    with self.lock:
                        if self.guard.stop.value:return
                        self.roundtrips.append(now-before)
                        if obs.t<=self.timeline.last_capture:
                            self.stats['discarded_out_of_order']+=1;continue
                        if self.timeline.window is None:
                            if self.warmups:
                                self.warmups-=1;self.stats['startup_warmup_calls']+=1;continue
                            held=self.live_state()
                            if np.max(abs(held[:6]-obs.state[:6]))>math.radians(1) or abs(held[6]-obs.state[6])>.005:
                                self.stats['bootstrap_discards']+=1;continue
                        first=self.timeline.window is None
                        report=self.timeline.publish(actions,obs.t,now)
                        self.guard.policy_time.value=now
                        self.stats['accepted_chunks']+=1;self.latencies.append(now-before);self.events.append(report)
                        if first:self.bootstrap_received=now;self.bootstrap.set()
                finally:
                    with self.lock:self.stats['inflight']-=1
        except BaseException as error:
            if not self.guard.stop.value:self.error=error;self.guard.stop.value=1


def run_remote_rtc(root,arm,cameras,bank,guard,client,previous,after,settings):
    def observation():
        cameras.check();obs=bank.observation()
        if obs.t<=after or np.any(obs.state_times<=after):raise NotReady('Waiting for post-preparation observation')
        return obs
    timeline=CaptureTimeline(previous,settings);smoother=CommandSmoother(previous,settings)
    producer=Producer(client,observation,lambda:bank.live()[0],timeline,guard,settings)
    guard.startup_deadline.value=time.monotonic()+settings['request_timeout_s']+3
    producer.thread.start();started=None;previous_time=time.monotonic();next_notice=0
    loops=0;max_dt=0.;error_record=None
    print('VPP2远程异步RTC：30Hz动作时间轴、120Hz控制，后台连续续块；Space/Esc/Ctrl+C 阻尼停止。',flush=True)
    try:
        while True:
            if producer.error:raise RuntimeError('VPP2 RTC inference failed: '+str(producer.error)) from producer.error
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();state,stamps=bank.live()
            check_state(state,settings)
            target=timeline.sample(now)
            if target is None:
                if started is not None:raise RuntimeError('VPP2 RTC prediction window exhausted')
                if now>guard.startup_deadline.value:raise RuntimeError('VPP2 RTC initial prediction timed out')
                arm.joints(previous[:6]);previous_time=now
            else:
                if started is None:
                    started=now;guard.policy_time.value=now;guard.running.value=1;previous_time=now
                dt=max(now-previous_time,1e-6);max_dt=max(max_dt,dt);loops+=1
                command=smoother.step(target,dt)
                arm.joints(command[:6]);arm.grip(command[6],settings['gripper_force_n'])
                previous=command;previous_time=now
                if now>=next_notice:
                    median=1000*np.median(producer.latencies[-20:]) if producer.latencies else 0
                    print('VPP2 RTC t=%.1fs chunks=%d RTT=%.0fms'%(now-started,timeline.accepted,median),flush=True)
                    next_notice=now+2
            time.sleep(max(0,1/settings['control_hz']-(time.monotonic()-now)))
    except BaseException as error:
        guard.stop.value=1;error_record=dict(type=type(error).__name__,message=str(error));raise
    finally:
        guard.stop.value=1
        client.close();producer.thread.join(.2)
        logs=Path(root)/'logs';logs.mkdir(exist_ok=True)
        record=dict(time=time.time(),model_name='vpp2',execution_mode='h100_async_rtc',
            policy_metadata=client.metadata,stats=producer.stats,events=list(producer.events),
            rtt_ms=[1000*x for x in producer.latencies],control_loops=loops,max_control_dt_s=max_dt,
            control_error=error_record,producer_error=None if producer.error is None else str(producer.error))
        content=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
        (logs/'last_policy_run.json').write_text(content)
        (logs/('policy_run_vpp2_rtc_'+time.strftime('%Y%m%d_%H%M%S')+'.json')).write_text(content)
