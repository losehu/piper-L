"""Capture-timed latent monitor; policy recovery events are separate from estop."""
from collections import deque
import math
import numpy as np


class RobustGate:
    def __init__(self,settings):
        self.settings=settings;self.errors=deque(maxlen=int(settings['window']))
        self.high=0;self.latched=False;self.cooldown_until=float('-inf')

    def update(self,error,stamp):
        c=self.settings
        if not math.isfinite(error+stamp):raise ValueError('Nonfinite monitor score/time')
        values=np.asarray(self.errors)
        median=float(np.median(values)) if len(values) else 0.
        mad=max(float(np.median(abs(values-median))) if len(values) else 0.,float(c['mad_floor']))
        on=max(float(c['floor']),median+c['on_mad']*mad)
        off=max(float(c['floor']),median+c['off_mad']*mad)
        event=False
        if len(values)>=c['warmup']:
            if self.latched:
                if error<off and stamp>=self.cooldown_until:self.latched=False
            elif stamp>=self.cooldown_until:
                self.high=self.high+1 if error>on else 0
                if self.high>=c['persistence']:
                    self.high=0;self.latched=True;self.cooldown_until=stamp+c['cooldown_s'];event=True
        # Freeze the reference during a sustained mismatch, so a failure does
        # not silently become the new normal threshold.
        if not self.latched:self.errors.append(float(error))
        return dict(error=float(error),threshold=on,release_threshold=off,event=event,latched=self.latched)


class LatentMonitor:
    def __init__(self,predict,k_step,settings):
        self.predict=predict;self.interval=k_step/30.;self.settings=settings
        self.gate=RobustGate(settings);self.previous=None;self.pending=deque();self.last_event=None
        self.stats=dict(observations=0,scores=0,events=0,stale_observations=0,unaligned_predictions=0,static_transitions=0)

    def observe(self,z,action,stamp):
        z=np.asarray(z,dtype=np.float32);action=np.asarray(action,dtype=np.float32)
        if z.ndim!=2 or action.shape!=(7,) or not np.isfinite(z).all() or not np.isfinite(action).all() or not math.isfinite(stamp):raise ValueError('Invalid latent/action/time')
        if self.previous is not None and stamp<=self.previous[0]:
            self.stats['stale_observations']+=1;return dict(event=False,stale=True)
        result=dict(event=False)
        if self.previous is not None:
            old_time,old_z=self.previous
            while self.pending and self.pending[0]['target']<=stamp:
                item=self.pending.popleft();target=item['target']
                if target<old_time-1e-6 or stamp-old_time>self.settings['max_interpolation_gap_s']:
                    self.stats['unaligned_predictions']+=1;continue
                fraction=(target-old_time)/(stamp-old_time)
                future=old_z+(z-old_z)*fraction;actual=future-item['z'];expected=item['predicted']
                a=actual.ravel();b=expected.ravel();an=float(np.linalg.norm(a));bn=float(np.linalg.norm(b))
                # No random noise is injected into static observations.
                if min(an,bn)<1e-4:
                    self.stats['static_transitions']+=1;continue
                score=1-float(np.dot(a,b)/(an*bn));check=self.gate.update(float(np.clip(score,0,2)),stamp)
                self.stats['scores']+=1
                result.update(check,source_at=item['captured'],target_at=target,interpolation_gap_s=stamp-old_time)
                if check['event']:
                    self.stats['events']+=1
                    self.last_event=dict(stamp=stamp,source_at=item['captured'],target_at=target,
                        error=check['error'],threshold=check['threshold'],target=(expected-actual).copy())
                    self.pending.clear();result['event']=True;break
        predicted=np.asarray(self.predict(z,action),dtype=np.float32)
        if predicted.shape!=z.shape or not np.isfinite(predicted).all():raise ValueError('Invalid latent prediction')
        self.pending.append(dict(captured=stamp,target=stamp+self.interval,z=z.copy(),predicted=predicted))
        while len(self.pending)>32:self.pending.popleft()
        self.previous=(float(stamp),z.copy());self.stats['observations']+=1
        return result
