"""30Hz RTC-Anything time axis with Kai0-style linear chunk transitions."""
import importlib.util
from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.smoothing import bridge_chunk,CriticallyDampedFilter


class ActionRejection(RuntimeError):
    def __init__(self,message,details):
        super().__init__(message)
        self.details=details


def validate_actions(actions, config, diagnostics=None):
    actions = np.asarray(actions,dtype=float).copy()
    horizon=int(config.get('action_horizon',50))
    if actions.shape != (horizon,7) or not np.isfinite(actions).all():
        raise ActionRejection('Policy must return finite %dx7 actions'%horizon,dict(shape=list(actions.shape)))
    prefix = int(config['execute_horizon'])
    lower,upper = np.asarray(config['policy_command_lower_rad']),np.asarray(config['policy_command_upper_rad'])
    projected=np.clip(actions[:,:6],lower,upper)
    correction=np.abs(projected-actions[:,:6])
    # Upstream buffer covers [cursor,cursor+chunk_size): frame `prefix`
    # is never executed, including by its separate lookahead buffer.
    relevant=correction[:prefix]
    indexes=np.argwhere(relevant>1e-10)
    entries=[dict(frame=int(frame),joint=int(joint+1),
                  predicted_deg=float(np.degrees(actions[frame,joint])),
                  command_deg=float(np.degrees(projected[frame,joint])),
                  correction_deg=float(np.degrees(relevant[frame,joint])),
                  lower_deg=float(np.degrees(lower[joint])),upper_deg=float(np.degrees(upper[joint])))
             for frame,joint in indexes]
    worst=max(entries,key=lambda item:item['correction_deg'],default=None)
    report=dict(corrected=bool(entries),corrected_values=len(entries),worst=worst,
                by_joint=np.count_nonzero(relevant>1e-10,axis=0).tolist(),
                max_correction_deg=np.degrees(relevant.max(axis=0)).tolist())
    if diagnostics is not None:diagnostics.update(report)
    actions[:,:6]=projected
    actions[:,6] = np.clip(actions[:,6],0,.1)
    return actions


class Timeline:
    def __init__(self, root, config, epoch, initial_action=None):
        path = Path(root)/'vendor/RTC-Anything/src/utils/real_time_chunking.py'
        spec = importlib.util.spec_from_file_location('upstream_rtc_anything',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.buffer = module.RealTimeChunkingBuffer(chunk_size=config['execute_horizon'])
        self.peek = module.RealTimeChunkingBuffer(chunk_size=config['execute_horizon'])
        self.old = module.RealTimeChunkingBuffer(chunk_size=config['execute_horizon'])
        self.config, self.epoch = config,float(epoch)
        self.initial_action=None if initial_action is None else np.array(initial_action,dtype=float,copy=True)
        self.has_prediction=False

    def enqueue(self, actions, capture_time, now):
        if not 0 <= now-capture_time <= self.config['max_observation_age_s']:
            raise RuntimeError('Policy response expired; no old chunk is executed')
        report={}
        actions = validate_actions(actions,self.config,report)
        cursor = int(round((capture_time-self.epoch)*30))
        if cursor < 0: raise RuntimeError('Observation precedes this rollout')
        with self.buffer.lock:
            self.old.chunks=dict(self.buffer.chunks)
            generation=self.buffer.generation
        bootstrap=not self.has_prediction and self.initial_action is not None
        old_action=(lambda _:self.initial_action) if bootstrap else self.old.get_action
        blended=0
        if self.config.get('smoothing_enabled',True):
            actions,blended=bridge_chunk(actions,cursor,(now-self.epoch)*30,old_action,
                                        self.config['chunk_smoothing_steps'],self.config['execute_horizon'])
        # A cubic tangent correction can overshoot its endpoints. The same full
        # firmware command bounds apply after the splice, before any execution.
        bounded=validate_actions(actions,self.config)
        report['bridge_clipped_values']=int(np.count_nonzero(abs(bounded-actions)>1e-10))
        actions=bounded
        # Replace atomically, with no empty queue or position-filter reset.
        with self.buffer.lock:
            if self.buffer.generation!=generation:raise RuntimeError('RTC generation changed during prediction')
            self.buffer.chunks={cursor:actions}
        self.has_prediction=True
        report['boundary_blended_frames']=blended
        return report

    def target(self, now):
        frame = (now-self.epoch)*30
        index = math.floor(frame)
        self.buffer.set_control_time(index)
        # Both interpolation endpoints must come from ONE immutable snapshot.
        # A producer swapping between two separate reads could otherwise mix
        # an old left endpoint and a new right endpoint in the same command.
        with self.buffer.lock:
            self.peek.chunks = dict(self.buffer.chunks)
            self.buffer.chunks={cursor:chunk for cursor,chunk in self.buffer.chunks.items()
                                if cursor+self.config['execute_horizon']>index}
        a = self.peek.get_action(index)
        b = self.peek.get_action(index+1)
        if a is None: return None
        return a if b is None else a+(b-a)*(frame-index)


class Client:
    def __init__(self, config):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        self.codec = msgpack_numpy
        self.ws = connect('ws://127.0.0.1:%d'%config['policy_port'],
                          compression=None,max_size=8*1024**2,open_timeout=3,
                          close_timeout=1,ping_interval=None)
        self.metadata = self.codec.unpackb(self.ws.recv(timeout=3))
        if (self.metadata.get('model')!='pi05_piper_grasp_%d'%config['model_step']
                or self.metadata.get('training_updates')!=config['model_step']
                or self.metadata.get('source_checkpoint_index')!=config['checkpoint_index']
                or self.metadata.get('action_horizon')!=50
                or self.metadata.get('physical_action_dim')!=7
                or self.metadata.get('fps')!=30 or not self.metadata.get('warmed_up')):
            self.ws.close()
            raise RuntimeError('Wrong policy identity/shape/rate or policy not warmed')
        self.request_id = 0

    def infer(self, observation, prompt):
        from openpi_client.image_tools import resize_with_pad
        self.request_id += 1
        payload = dict(state=observation.state.astype(np.float32),prompt=prompt,
                       request_id=self.request_id,observed_at=observation.t)
        for key,image in observation.images.items():
            payload[key] = resize_with_pad(image,224,224)
        self.ws.send(self.codec.packb(payload))
        raw = self.ws.recv(timeout=2)
        if isinstance(raw,str): raise RuntimeError(raw)
        response = self.codec.unpackb(raw)
        if (response['request_id']!=self.request_id
                or abs(response['observed_at']-observation.t)>1e-6):
            raise RuntimeError('Response belongs to another observation')
        return response['actions']

    def close(self):
        self.ws.close()


class Producer:
    def __init__(self, client, observation, timeline, guard, prompt):
        self.client,self.observation,self.timeline,self.guard,self.prompt = client,observation,timeline,guard,prompt
        self.error = None
        self.latencies = []
        self.error_details=None
        self.failure_observation=self.failure_actions=None
        self.events=deque(maxlen=256)
        self.stats=dict(requests=0,accepted_chunks=0,projected_chunks=0,
                        projected_values_by_joint=[0]*6,max_correction_deg=[0.]*6,
                        observation_waits=0,max_observation_wait_s=0.,
                        feedback_max_overshoot_deg=[0.]*6,feedback_outside_command_samples=[0]*6,
                        boundary_smoothed_chunks=0)
        self.observation_wait=None
        self.capture_latencies=[]
        self.prediction_history=deque(maxlen=1200)
        self.thread = threading.Thread(target=self.run,daemon=True)

    def run(self):
        last_capture = float('-inf')
        obs=actions=None
        next_notice=0
        waiting_since=None
        try:
            while not self.guard.stop.value:
                obs=actions=None
                try:obs = self.observation()
                except NotReady as pending:
                    now=time.monotonic()
                    if waiting_since is None:waiting_since=now
                    duration=now-waiting_since
                    self.stats['observation_waits']+=1
                    self.stats['max_observation_wait_s']=max(self.stats['max_observation_wait_s'],duration)
                    self.observation_wait=dict(reason=str(pending),details=pending.details,
                                               started=waiting_since,duration_s=duration,recovered=False)
                    # Neither a chunk nor the policy heartbeat is refreshed.
                    # Existing RTC expiry / watchdogs still stop a real loss.
                    time.sleep(.005)
                    continue
                if obs.t <= last_capture:
                    time.sleep(.005)
                    continue
                if waiting_since is not None:
                    duration=time.monotonic()-waiting_since
                    self.stats['max_observation_wait_s']=max(self.stats['max_observation_wait_s'],duration)
                    self.observation_wait.update(duration_s=duration,recovered=True)
                    waiting_since=None
                last_capture = obs.t
                before = time.monotonic()
                self.stats['requests']+=1
                actions = self.client.infer(obs,self.prompt)
                now = time.monotonic()
                if self.guard.stop.value: return
                report=self.timeline.enqueue(actions,obs.t,now)
                self.prediction_history.append((obs.t,now,np.array(obs.state,copy=True),
                                                np.array(actions,dtype=np.float32,copy=True)))
                self.stats['accepted_chunks']+=1
                if report.get('boundary_blended_frames',0):self.stats['boundary_smoothed_chunks']+=1
                if report['corrected']:
                    self.stats['projected_chunks']+=1
                    self.stats['projected_values_by_joint']=(np.asarray(self.stats['projected_values_by_joint'])+report['by_joint']).tolist()
                    self.stats['max_correction_deg']=np.maximum(self.stats['max_correction_deg'],report['max_correction_deg']).tolist()
                    self.events.append(dict(capture_time=obs.t,**report))
                    if now>=next_notice:
                        item=report['worst']
                        print('固件限位裁剪：J%d 第%d帧 %.2f° → %.2f°。'%(
                            item['joint'],item['frame'],item['predicted_deg'],item['command_deg']),flush=True)
                        next_notice=now+2
                self.guard.policy_time.value = now
                self.latencies.append(now-before)
                self.capture_latencies.append(now-obs.t)
        except BaseException as error:
            self.guard.stop.value = 1  # Stop first; only then preserve diagnostics.
            self.error = str(error)
            self.error_details=getattr(error,'details',None)
            self.failure_observation,self.failure_actions=obs,actions

    def record_feedback(self,state,settings):
        q=np.asarray(state[:6]);low=np.asarray(settings['policy_command_lower_rad']);high=np.asarray(settings['policy_command_upper_rad'])
        excess=np.maximum(np.maximum(low-q,q-high),0)
        self.stats['feedback_max_overshoot_deg']=np.maximum(self.stats['feedback_max_overshoot_deg'],np.degrees(excess)).tolist()
        self.stats['feedback_outside_command_samples']=(np.asarray(self.stats['feedback_outside_command_samples'])+(excess>1e-8)).tolist()

    def save_report(self, root, control_error=None):
        root=Path(root)/'logs';root.mkdir(exist_ok=True)
        stamp=time.strftime('%Y%m%d_%H%M%S')
        record=dict(time=time.time(),stats=self.stats,
                    error=self.error or (control_error or {}).get('message'),
                    error_origin='policy' if self.error else ('controller' if control_error else None),
                    error_details=self.error_details,control_error=control_error,
                    observation_wait=self.observation_wait,
                    recent_projections=list(self.events),rejected_observation=None)
        record['timing']={}
        record['execution_settings']={key:self.timeline.config[key] for key in
            ['fps','control_hz','execute_horizon','chunk_smoothing_method',
             'chunk_smoothing_steps','command_filter_tau_s','model_step','checkpoint_index','checkpoint']}
        record['policy_metadata']=getattr(self.client,'metadata',{})
        for name,values in [('request_rtt',self.latencies),('capture_to_action',self.capture_latencies)]:
            if values:
                record['timing'][name]=dict(p50_ms=float(1000*np.median(values)),
                    p95_ms=float(1000*np.percentile(values,95)),p99_ms=float(1000*np.percentile(values,99)))
        try:
            rows=list(self.prediction_history)
            if rows:
                history=root/('policy_chunks_'+stamp+'.npz')
                np.savez_compressed(history,capture_times=np.array([r[0] for r in rows]),
                    arrival_times=np.array([r[1] for r in rows]),states=np.array([r[2] for r in rows]),
                    raw_actions=np.array([r[3] for r in rows]))
                record['prediction_history']=str(history)
            if self.error and self.failure_observation is not None:
                obs=self.failure_observation
                payload=dict(state=obs.state,observed_at=obs.t,state_times=obs.state_times,
                             prompt=np.array(self.prompt),**obs.images)
                if self.failure_actions is not None:payload['raw_actions']=self.failure_actions
                path=root/('policy_rejected_'+stamp+'.npz')
                np.savez_compressed(path,**payload)
                record['rejected_observation']=str(path)
            content=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
            (root/('policy_run_'+stamp+'.json')).write_text(content)
            (root/'last_policy_run.json').write_text(content)
            print('策略诊断记录：',root/'last_policy_run.json',flush=True)
        except (OSError,ValueError) as error:print('策略诊断保存失败：',error,flush=True)


def limited_target(target, previous, dt, config):
    target,previous = np.asarray(target),np.asarray(previous)
    if (target.shape!=(7,) or previous.shape!=(7,) or not np.isfinite(target).all()
            or not 0 < dt <= .1):
        raise RuntimeError('Invalid command or control-loop gap over 100ms')
    lower,upper=np.asarray(config['policy_command_lower_rad']),np.asarray(config['policy_command_upper_rad'])
    if np.any(previous[:6]<lower-1e-8) or np.any(previous[:6]>upper+1e-8):
        raise RuntimeError('Previous policy command outside configured command bounds')
    target=target.copy();target[:6]=np.clip(target[:6],lower,upper);target[6]=np.clip(target[6],0,.1)
    speeds = np.r_[np.radians(config['policy_joint_speed_deg_s']),config['gripper_speed_m_s']]
    command=previous+np.clip(target-previous,-speeds*dt,speeds*dt)
    command[6]=np.clip(command[6],0,.1)
    return command


class CommandSmoother:
    """Slew-limited reference followed by a persistent critically damped filter."""
    def __init__(self, initial, config):
        self.reference=np.array(initial,dtype=float,copy=True)
        self.config=config
        self.filter=CriticallyDampedFilter(self.reference,config['command_filter_tau_s'])

    def step(self,target,dt):
        # Keep the reference's state separate from the filtered actual output.
        # Reusing the filtered position as the slew limiter origin each time
        # would accidentally slow every motion almost to a standstill.
        self.reference=limited_target(target,self.reference,dt,self.config)
        if not self.config.get('smoothing_enabled',True):return self.reference.copy()
        command=self.filter.step(self.reference,dt)
        command[:6]=np.clip(command[:6],self.config['policy_command_lower_rad'],
                            self.config['policy_command_upper_rad'])
        command[6]=np.clip(command[6],0,.1)
        return command
