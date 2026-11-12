"""Kai0-style old-to-new linear chunk transition, adapted to a timed 7D stream.

Based on StreamActionBuffer.integrate_new_chunk in OpenDriveLab/kai0,
commit 9d93078c757840f50e75248c5c5a94ab7b41e13a (Apache-2.0).
The original source and license are in vendor/kai0_reference.
Unlike the ROS queue, this adapter aligns by observation timestamps and never
pads or extends an expired old action block.
"""
import math
import numpy as np


def blend_chunk(actions, cursor, current_frame, old_action, steps, horizon):
    result=np.array(actions,dtype=float,copy=True)
    start=max(cursor,math.ceil(current_frame))
    count=0
    for frame in range(cursor,min(start,cursor+horizon)):
        old=old_action(frame)
        if old is None:continue
        result[frame-cursor]=old
        count+=1
    # Always end the available overlap at 100% new. A short remaining old
    # prefix must not end with nonzero old weight and then snap to the new plan.
    future=[]
    for frame in range(start,min(cursor+horizon,start+steps)):
        old=old_action(frame)
        if old is None:break
        future.append((frame,np.asarray(old)))
    overlap=len(future)
    for i,(frame,old) in enumerate(future):
        weight=i/(overlap-1) if overlap>1 else 0.
        result[frame-cursor]=(1-weight)*old+weight*result[frame-cursor]
        count+=1
    return result,count


def bridge_chunk(actions,cursor,current_frame,old_action,steps,horizon):
    """Join to a new forecast while retaining the old position and tangent.

    Linear position blending adds a velocity term proportional to the position
    disagreement. A cubic correction also matches the outgoing tangent, then
    decays both position and tangent corrections to zero over valid overlap.
    """
    result=np.array(actions,dtype=float,copy=True)
    start=max(cursor,math.ceil(current_frame))
    samples={}
    for frame in range(cursor,min(cursor+horizon,start+steps)):
        old=old_action(frame)
        if old is not None:samples[frame]=np.array(old,dtype=float,copy=True)
    if start not in samples:return result,0
    for frame in range(cursor,min(cursor+horizon,start+2)):
        if frame in samples:result[frame-cursor]=samples[frame]
    anchor=start+1
    available=[]
    for frame in range(anchor,min(cursor+horizon,start+steps)):
        if frame not in samples:break
        available.append(frame)
    if len(available)<2:
        return blend_chunk(actions,cursor,current_frame,samples.get,steps,horizon)
    end=available[-1];span=end-anchor
    offset=samples[anchor]-actions[anchor-cursor]
    old_velocity=samples[anchor]-samples[start]
    new_velocity=actions[anchor+1-cursor]-actions[anchor-cursor]
    velocity_offset=old_velocity-new_velocity
    for frame in range(anchor,end+1):
        s=(frame-anchor)/span
        result[frame-cursor]=(actions[frame-cursor]+(2*s**3-3*s**2+1)*offset
                             +(s**3-2*s**2+s)*span*velocity_offset)
    return result,len(samples)


class CriticallyDampedFilter:
    """Exact two-pole position filter for irregular control-loop intervals.

    Its three nonnegative weights sum to one, so bounded inputs and initial
    states remain bounded without extrapolation or resetting at chunk arrivals.
    The low-frequency ramp lag is 2*tau. Emergency stopping bypasses this filter.
    """
    def __init__(self, initial, tau):
        self.first=np.array(initial,dtype=float,copy=True)
        self.second=self.first.copy()
        self.tau=float(tau)

    def step(self, target, dt):
        target=np.asarray(target,dtype=float)
        if (target.shape!=self.first.shape or not np.isfinite(target).all()
                or not 0<dt<=.1):
            raise RuntimeError('Invalid smoothing input or control-loop gap over 100ms')
        h=dt/self.tau
        decay=math.exp(-h)
        previous_first=self.first.copy()
        self.first=(1-decay)*target+decay*previous_first
        self.second=(1-decay*(1+h))*target+decay*self.second+h*decay*previous_first
        return self.second.copy()
