"""Trim handover waits on the original clock; never stitch across removed time.

Only the edges of control generations are trimmed. Interior task pauses return
when later motion is observed. An unfinished live tail is withheld, so a future
handover cannot turn already published long waits into imitation targets.
"""
import numpy as np
from rlt_piper.contract import CHUNK

VERSION='handover-edges-v1'
CONFIG=dict(joint_motion_deg=.05,gripper_motion_m=.0003,motion_window_s=.25,
            edge_context_s=.25,minimum_idle_s=.75)

def _different(values,reference):
    delta=np.abs(values-reference)
    return np.any(delta[:,:6]>np.radians(CONFIG['joint_motion_deg']),axis=1)|(delta[:,6]>CONFIG['gripper_motion_m'])

def keep_commands(data,meta):
    times=data['command_t'];commands=data['commands'];sources=data['command_source'];generations=data['command_generation']
    keep=~np.isin(sources,[4,5]);events=meta.get('events',[]);ranges=[]
    state=data.get('command_state');all_state=state is not None
    for gen in np.unique(generations):
        ix=np.flatnonzero((generations==gen)&np.isin(sources,[0,1,2]))
        if not len(ix):continue
        t=times[ix];q=commands[ix]
        entry=next((e for e in events if e['generation']==int(gen) and e['mode'] in ('human','policy')),None)
        later=[e for e in events if e['generation']>int(gen) and e['t']>=t[0] and e['mode'] in ('aligning','human','policy')]
        exit_event=min(later,key=lambda e:e['t']) if later else None
        if entry is None and exit_event is None:continue  # Historical records without handover markers.
        previous=np.searchsorted(t,t-CONFIG['motion_window_s'],side='right')-1;previous=np.maximum(previous,0)
        moving=_different(q,q[previous]);from_start=_different(q,q[0]);from_end=_different(q,q[-1])
        if all_state:
            measured=state[ix]
            # Old recordings may have no measured state. A missing state must
            # not hide command intent, nor be treated as zero robot feedback.
            good=np.isfinite(measured).all(axis=1);paired=good&good[previous]
            if good[0]:from_start |= good&_different(measured,measured[0])
            if good[-1]:from_end |= good&_different(measured,measured[-1])
            moving |= paired&_different(measured,measured[previous])
        first=np.flatnonzero(moving|from_start);last=np.flatnonzero(moving|from_end)
        lo,hi=float(t[0]),float(t[-1]);reason=[]
        # Unlabelled and live streams have no terminal truth yet. Hold the
        # open trailing pause until motion resumes or the episode is labelled.
        live_tail=bool(meta.get('live')) and exit_event is None
        if not len(first) or not len(last):
            finished=bool(meta.get('labelled') and meta.get('terminal_available') and not meta.get('error'))
            if finished and exit_event is None:
                # Preserve one real terminal context for S/F instead of losing
                # the task reward, even when the final post-return plan held still.
                terminal_end=min(float(meta.get('label_at') or t[-1]),float(t[-1]))
                lo=max(lo,terminal_end-CHUNK/30);reason.append('terminal_outcome_context')
            elif live_tail or entry is not None or exit_event is not None:
                keep[ix]=False;reason.append('no_motion_yet')
                ranges.append(dict(generation=int(gen),start=None,end=None,reasons=reason));continue
        else:
            context=CONFIG['edge_context_s'];minimum=CONFIG['minimum_idle_s']
            if entry is not None and t[first[0]]-t[0]>=minimum:
                lo=float(max(t[0],t[first[0]]-context));reason.append('handover_leading_wait')
            if exit_event is not None and t[-1]-t[last[-1]]>=minimum:
                hi=float(min(t[-1],t[last[-1]]+context));reason.append('handover_trailing_wait')
            elif live_tail:
                hi=float(min(t[-1],t[last[-1]]+context));reason.append('pending_live_tail')
        keep[ix]&=(t>=lo-1e-9)&(t<=hi+1e-9)
        ranges.append(dict(generation=int(gen),start=lo,end=hi,reasons=reason))
    # Duration is a diagnostic on actual command time, not synthetic frame count.
    dt=np.r_[np.clip(np.diff(times),0,.1),0.] if len(times) else np.array([])
    report=dict(version=VERSION,config=CONFIG,ranges=ranges,excluded_seconds=float(np.sum(dt[~keep])),
                excluded_commands=int(np.count_nonzero(~keep)),raw_commands=len(times),clock='original_monotonic_no_concatenation')
    return keep,report
