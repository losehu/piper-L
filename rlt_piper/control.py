"""Single TX authority for RTC policy, relative leader takeover and fresh return."""
from collections import deque
import threading,time,numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.policy import Timeline,CommandSmoother
from rtc_piper.control import check_state

def run(root,arm,cameras,bank,guard,policy,previous,after,settings,recorder,handover=None,leader=None,ui=None):
    from rlt_piper.hil import Handover
    h=handover or Handover();h.after=after
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True);arm.joints(previous[:6])
        try:first=bank.observation()
        except NotReady:time.sleep(1/120);continue
        if first.t>after and np.all(first.state_times>after):break
        time.sleep(1/120)
    timeline=Timeline(root,settings,first.t,initial_action=previous);smoother=CommandSmoother(previous,settings)
    stop=threading.Event();errors=[];rtts=deque(maxlen=100);active_source='hold';learner_status={}
    def infer():
        nonlocal active_source,learner_status
        last=float('-inf')
        try:
            while not stop.is_set() and not guard.stop.value:
                with h.lock:gen,mode,after=h.generation,h.mode,h.after
                try:
                    obs=bank.observation()
                    if obs.t<=max(last,after) or np.any(obs.state_times<=after):raise NotReady('No fresh post-handover observation')
                except NotReady:time.sleep(.005);continue
                last=obs.t;before=time.monotonic()
                result=policy.request('features' if mode=='human' else 'infer',obs,generation=gen,prompt=settings['prompt']);now=time.monotonic()
                with h.lock:
                    if stop.is_set() or guard.stop.value:return
                    if not h.accepts(gen,obs.t):continue
                    if mode=='policy':timeline.enqueue(result['actions'],obs.t,now);active_source=result['rlt']['source']
                    recorder.plan(obs,result,now,source='human' if mode=='human' else None)
                    guard.policy_time.value=now;rtts.append(now-before);learner_status=result.get('learner',{})
        except BaseException as e:
            if not stop.is_set():errors.append(repr(e));guard.stop.value=1
    thread=threading.Thread(target=infer,daemon=True);thread.start()
    started=False;previous_time=time.monotonic();next_report=0;next_ui=0
    guard.startup_deadline.value=time.monotonic()+3
    try:
        while True:
            if errors:raise RuntimeError('RLT worker failed: '+errors[0])
            if recorder.error:raise RuntimeError(recorder.error)
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();measured,stamps=bank.live();check_state(measured,settings)
            raw_leader=leader_times=None
            with h.lock:
                if h.requested:
                    if leader is None:raise RuntimeError('Human takeover requires a verified leader reader')
                    mode=h.switch(previous,leader,now)
                    # Replace all RTC buffers, and cancel the outgoing filter's residual velocity.
                    timeline=Timeline(root,settings,now,initial_action=previous);smoother=CommandSmoother(previous,settings)
                    active_source='hold';started=False
                    guard.running.value=0;guard.startup_deadline.value=now+3
                    recorder.event(mode,h.generation,now)
                    print('人工接管（相对当前位置）' if mode=='human' else '交还策略：保持当前位置，等待新观测和新chunk',flush=True)
                if h.mode=='human':
                    target,raw_leader,leader_times=h.target(leader);source='human'
                    # Policy watchdog remains active: features still arrive during takeover.
                    if not started:started=True;guard.running.value=1;guard.policy_time.value=now
                else:
                    target=timeline.target(now)
                    if target is None:
                        if started:raise RuntimeError('RLT RTC action prefix exhausted')
                        target=previous;source='hold'
                    else:
                        source=active_source
                        if not started:started=True;guard.running.value=1;guard.policy_time.value=now
                command=smoother.step(target,max(now-previous_time,1e-6));guard.check()
                arm.joints(command[:6])
                if started:arm.grip(command[6],settings['gripper_force_n'])
                # SDK uses millidegree joint integers and micrometre gripper integers.
                actual=np.r_[np.rint(np.degrees(command[:6])*1000)*np.pi/180000,np.rint(command[6]*1e6)/1e6]
                recorder.command(time.monotonic(),actual,source,h.generation,raw_leader,leader_times)
            previous=command;previous_time=now
            if ui and now>=next_ui:
                ui.update(phase='running',source='human' if h.mode=='human' else active_source,
                    rtt_ms=float(1000*np.median(rtts)) if rtts else None,learner=learner_status,
                    joint_deg=np.degrees(measured[:6]).tolist(),gripper_mm=float(measured[6]*1000))
                next_ui=now+.1
            if now>=next_report:
                print('RLT + RTC + 平滑；%s；RTT=%.0fms；训练=%s step=%s；I接管/交还 S成功/F失败 Space/Esc/Ctrl+C阻尼停止。'%(
                    h.mode+'/'+active_source,1000*np.median(rtts) if rtts else 0,learner_status.get('status','off'),learner_status.get('global_step',0)),flush=True)
                next_report=now+2
            time.sleep(max(0,1/120-(time.monotonic()-now)))
    finally:
        guard.stop.value=1;stop.set();policy.close();thread.join(2)
