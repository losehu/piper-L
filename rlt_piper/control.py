"""Existing RTC and output smoothing with native RLT recording; no Corrector."""
from collections import deque
import threading,time,numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.policy import Timeline,CommandSmoother
from rtc_piper.control import check_state

def run(root,arm,cameras,bank,guard,policy,previous,after,settings,recorder):
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True);arm.joints(previous[:6])
        try:first=bank.observation()
        except NotReady:time.sleep(1/120);continue
        if first.t>after and np.all(first.state_times>after):break
        time.sleep(1/120)
    timeline=Timeline(root,settings,first.t,initial_action=previous);smoother=CommandSmoother(previous,settings)
    stop=threading.Event();errors=[];rtts=deque(maxlen=100)
    def infer():
        last=float('-inf')
        try:
            while not stop.is_set() and not guard.stop.value:
                try:
                    obs=bank.observation()
                    if obs.t<=last:raise NotReady('No new RLT observation')
                except NotReady:time.sleep(.005);continue
                last=obs.t;before=time.monotonic();result=policy.request('infer',obs,prompt=settings['prompt']);now=time.monotonic()
                if stop.is_set() or guard.stop.value:return
                timeline.enqueue(result['actions'],obs.t,now);recorder.plan(obs,result,now)
                guard.policy_time.value=now;rtts.append(now-before)
        except BaseException as e:
            if not stop.is_set():errors.append(repr(e));guard.stop.value=1
    thread=threading.Thread(target=infer,daemon=True);thread.start()
    started=False;previous_time=time.monotonic();next_report=0
    guard.startup_deadline.value=time.monotonic()+3
    try:
        while True:
            if errors:raise RuntimeError('RLT worker failed: '+errors[0])
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();measured,stamps=bank.live();check_state(measured,settings)
            target=timeline.target(now)
            if target is None:
                if started:raise RuntimeError('RLT RTC action prefix exhausted')
                target=previous
            elif not started:started=True;guard.running.value=1;guard.policy_time.value=now
            command=smoother.step(target,max(now-previous_time,1e-6));guard.check()
            arm.joints(command[:6])
            if started:arm.grip(command[6],settings['gripper_force_n'])
            recorder.command(time.monotonic(),command);previous=command;previous_time=now
            if now>=next_report:
                print('RLT + RTC + 平滑，Corrector关闭；RTT=%.0fms；S成功/F失败，Space/Esc/Ctrl+C阻尼停止。'%(1000*np.median(rtts) if rtts else 0),flush=True)
                next_report=now+2
            time.sleep(max(0,1/120-(time.monotonic()-now)))
    finally:
        guard.stop.value=1;stop.set();policy.close();thread.join(.5)
