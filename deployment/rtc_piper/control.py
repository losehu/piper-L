"""Automatic command-line deployment: official SDK + timed RTC action stream."""
import math
import json
from pathlib import Path
import time

import numpy as np
from rtc_piper.observations import NotReady
from rtc_piper.hardware import JointCommunicationFault
from rtc_piper.policy import Timeline, Producer, CommandSmoother


def check_state(state, config):
    state = np.asarray(state,dtype=float)
    if state.shape!=(7,) or not np.isfinite(state).all():
        raise RuntimeError('Invalid measured state')
    # These are measured coordinates, not commanded targets. Keep encoder
    # overshoot unchanged; firmware angle/driver faults are checked by status().
    return state


def choose_start(state, starts, config):
    state = check_state(state,config)
    starts = np.asarray(starts,dtype=float)
    if starts.ndim!=2 or starts.shape[1]!=7 or not np.isfinite(starts).all():
        raise ValueError('Invalid demonstration starts')
    delta = np.abs(np.degrees(starts[:,:6]-state[None,:6]))
    row = int(np.argmin(np.max(delta,axis=1)))
    target=np.clip(starts[row,:6],config['policy_command_lower_rad'],config['policy_command_upper_rad'])
    return row,check_state(np.r_[target,np.clip(state[6],0,.1)],config)


def trajectory(start, goal, elapsed):
    """Synchronized trapezoid, 5°/s and 10°/s² before CAN quantization."""
    distance = float(np.max(np.abs(np.degrees(goal[:6]-start[:6]))))
    if distance < 1e-9: return goal.copy(),0.0
    ramp = min(.5,math.sqrt(distance/10))
    peak = 10*ramp
    coast = max(0,distance/peak-ramp)
    duration = 2*ramp+coast
    t = min(max(elapsed,0),duration)
    if t<ramp: travelled = 5*t*t
    elif t<ramp+coast: travelled=5*ramp*ramp+peak*(t-ramp)
    else: travelled=distance-5*(duration-t)**2
    out=start.copy();out[:6]+=min(1,travelled/distance)*(goal[:6]-start[:6])
    return out,duration


class Settling:
    def __init__(self):
        self.last = None
        self.since = None
        self.speed = np.full(3,np.inf)

    def update(self, state, stamps, position_ok=True):
        if self.last is None:
            self.last = (state.copy(),stamps.copy())
            return False
        previous,times = self.last
        for pair in range(3):
            dt=stamps[pair]-times[pair]
            if dt<0: raise RuntimeError('Feedback clock moved backwards')
            if dt>0:
                section=slice(pair*2,pair*2+2)
                self.speed[pair]=np.max(np.abs(np.degrees(state[section]-previous[section])))/dt
                previous[section]=state[section];times[pair]=stamps[pair]
        watermark=float(np.min(stamps[:3]))
        if position_ok and np.all(self.speed<=1):
            if self.since is None: self.since=watermark
            return watermark-self.since>=.3
        self.since=None
        return False


def wait_ready(arm, cameras, bank, guard, seconds=8):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        guard.check();cameras.check()
        try:
            state,stamps=bank.live()
            code,enabled=arm.status(allow_estop=True)
            observation=bank.observation()
            return state,stamps,code,enabled,observation
        except NotReady:
            time.sleep(.01)
    raise RuntimeError('Timed out waiting for measured state and synchronized cameras')


def wait_after_reset(arm, cameras, bank, guard, wall, mono, config, trace):
    """After Reset/FC role setup, await 300ms of fresh healthy feedback.

    Reset/role setup can briefly interrupt joint communication. A reported pure
    communication error may clear within the existing 3s recovery window; no
    protection, driver fault, runtime fault or persistent error is bypassed.
    """
    label=trace.get('stage','复位')
    deadline = time.monotonic()+3
    fresh_after, stable_since, last = wall, None, '等待复位后的完整新反馈'
    signature = None
    while time.monotonic()<deadline:
        guard.check();cameras.check()
        try:
            code,enabled=arm.status(allow_estop=True,after=fresh_after)
            state,stamps=bank.live();check_state(state,config)
            if code==0 and np.all(stamps>mono):
                now=time.monotonic()
                if stable_since is None:
                    stable_since=now
                    trace['events'].append(dict(elapsed_s=now-mono,event='healthy_feedback'))
                if now-stable_since>=.3:
                    trace.update(status='passed',elapsed_s=now-mono)
                    print(label+'完成：全部反馈正常并稳定300ms。',flush=True)
                    return state
            else:
                stable_since=None;last='固件仍处于急停/复位状态'
        except JointCommunicationFault as error:
            stable_since=None
            fresh_after=time.time()  # Each driver must update after this fault.
            details=error.details
            last='关节通信异常，关节=%s，0x2A1=%s'%(details['communication_joints'],details['status_can_hex'])
            new_signature=(details['arm_status'],details['err_code'])
            if new_signature!=signature:
                signature=new_signature
                trace['events'].append(dict(elapsed_s=time.monotonic()-mono,event='communication_fault',**details))
                print(label+'等待：'+last+'；暂不使能或发送目标。',flush=True)
        except NotReady as error:
            stable_since=None;last=str(error)
        time.sleep(.01)
    trace.update(status='failed',reason=last,elapsed_s=time.monotonic()-mono)
    raise RuntimeError(label+'后3秒内未取得稳定正常反馈：'+last+'；未发送模式/关节目标/使能，请检查对应关节连接及供电')


def _write_recovery(trace):
    filename='role_recovery.json' if trace.get('stage')=='从臂角色设置' else 'reset_recovery.json'
    path=Path(__file__).resolve().parents[1]/'logs'/filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')


def confirm_policy_hold(arm, cameras, bank, guard, hold, config):
    """Switch only after alignment; any protection during the switch is fatal."""
    arm.set_control_profile('policy')
    arm.joints(hold[:6])
    wall,mono=time.time(),time.monotonic()
    settling=Settling()
    until=mono+2
    while time.monotonic()<until:
        guard.check();cameras.check()
        state,stamps=bank.live();check_state(state,config)
        try:
            arm.status(require_enabled=True,after=wall)
            bank.observation()
            if np.all(stamps>mono) and settling.update(state,stamps):
                print('高跟随保持正常并稳定；正在请求新策略。',flush=True)
                return time.monotonic(),hold
        except NotReady:pass
        arm.joints(hold[:6]);time.sleep(1/120)
    raise RuntimeError('High-follow hold did not become freshly stable within 2s')


def prepare(arm, cameras, bank, guard, starts, config):
    state,stamps=bank.live();check_state(state,config)
    _,goal=choose_start(state,starts,config)
    bounded_state=state.copy()
    bounded_state[:6]=np.clip(state[:6],config['policy_command_lower_rad'],config['policy_command_upper_rad'])
    _,motion_time=trajectory(bounded_state,goal,0)
    guard.arm(startup_seconds=motion_time+15)
    code,enabled=arm.status(allow_estop=True)
    reset=False
    if code==1:
        print('恢复固件急停：请保持可靠支撑。',flush=True)
        wall,mono=time.time(),time.monotonic()
        arm.arm.reset();reset=True
        trace=dict(status='waiting',reset_wall_time=wall,events=[])
        try:state=wait_after_reset(arm,cameras,bank,guard,wall,mono,config,trace)
        except BaseException as error:
            trace.update(status='failed',reason=str(error))
            raise
        finally:_write_recovery(trace)
    elif any(enabled):
        raise RuntimeError('Already enabled outside this session; refuse takeover')
    state=check_state(state,config)
    # Some V189 firmware keeps driver-enable bits across Reset. This session
    # owns the explicit recovery; overwrite its target with measured hold before
    # any preparation motion. Normal already-enabled arms are never taken over.
    if not reset and any(enabled): raise RuntimeError('Unexpected enabled arm')
    # The working collector first restores FC role, then enables in ordinary
    # MOVEJ at 5%. It enters high following only after its preparation move.
    print('设置运动输出从臂角色；先以普通MOVEJ保持当前位置。',flush=True)
    wall,mono=time.time(),time.monotonic()
    arm.configure_follower()
    trace=dict(status='waiting',stage='从臂角色设置',setup_wall_time=wall,events=[])
    try:state=wait_after_reset(arm,cameras,bank,guard,wall,mono,config,trace)
    except BaseException as error:
        trace.update(status='failed',reason=str(error));raise
    finally:_write_recovery(trace)
    arm.set_control_profile('prepare')
    hold=state.copy()
    hold[:6]=np.clip(hold[:6],config['policy_command_lower_rad'],config['policy_command_upper_rad'])
    hold[6]=np.clip(hold[6],0,.1)
    arm.joints(hold[:6])
    enabled_after=time.time()
    until=time.monotonic()+2
    next_enable=0
    while True:
        guard.check();cameras.check()
        now=time.monotonic()
        arm.joints(hold[:6])  # Keep the measured hold streaming while enabling.
        if now>=next_enable:
            arm.arm.enable();next_enable=now+.2
        try:
            code,enabled=arm.status(require_enabled=False,after=enabled_after)
            packet=arm.arm.get_arm_status()
            if all(enabled) and int(packet.msg.ctrl_mode)==1 and int(packet.msg.mode_feedback)==1:break
        except NotReady:pass
        if now>until:raise RuntimeError('Fresh CAN/MOVEJ enable confirmation timed out')
        time.sleep(max(0,1/120-(time.monotonic()-now)))
    print('已确认保持当前姿态并使能；准备阶段夹爪不动作。',flush=True)
    settling=Settling()
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True)
        state,stamps=bank.live();check_state(state,config)
        if settling.update(state,stamps):break
        arm.joints(hold[:6]);time.sleep(1/120)
    row,target=choose_start(state,starts,config)
    start=hold.copy();begun=time.monotonic();settling=Settling();previous=hold.copy();previous_time=begun
    print('低速对齐示范起点 %d，目标角度 %s，最大5°/s。'%(row,np.round(np.degrees(target[:6]),2)),flush=True)
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True)
        now=time.monotonic();state,stamps=bank.live();check_state(state,config)
        dt=now-previous_time
        if not 0<=dt<=.1:raise RuntimeError('Preparation control gap exceeds 100ms')
        try:bank.observation()
        except NotReady:
            # Pause preparation at its last target rather than advance blind.
            arm.joints(previous[:6]);previous_time=now;time.sleep(1/120);continue
        desired,duration=trajectory(start,target,now-begun)
        step=math.radians(5)*max(dt,1e-6)
        desired[:6]=previous[:6]+np.clip(desired[:6]-previous[:6],-step,step)
        arm.joints(desired[:6]);previous=desired;previous_time=now
        position_ok=now-begun>=duration and np.max(np.abs(np.degrees(state[:6]-target[:6])))<=1
        if settling.update(state,stamps,position_ok):
            print('示范起点到位且已稳定，切换高跟随并保持。',flush=True)
            return confirm_policy_hold(arm,cameras,bank,guard,previous,config)
        time.sleep(max(0,1/120-(time.monotonic()-now)))


def run_policy(root, arm, cameras, bank, guard, client, previous, after, config):
    # Entire capture and all four feedback groups must be from this new stage.
    while True:
        guard.check();cameras.check();arm.status(require_enabled=True)
        live,live_stamps=bank.live();check_state(live,config)
        arm.joints(previous[:6])
        try:obs=bank.observation()
        except NotReady:
            time.sleep(1/120);continue
        if obs.t>after and np.all(obs.state_times>after):break
        time.sleep(1/120)
    timeline=Timeline(root,config,obs.t,initial_action=previous)
    smoother=CommandSmoother(previous,config)
    def observation():
        cameras.check()
        return bank.observation()
    producer=Producer(client,observation,timeline,guard,config['prompt'])
    producer.thread.start()
    started=None;previous_time=time.monotonic();next_report=0
    state,stamps=live,live_stamps
    control_error=None
    try:
        while True:
            guard.check();cameras.check();arm.status(require_enabled=True)
            now=time.monotonic();state,stamps=bank.live();check_state(state,config)
            producer.record_feedback(state,config)
            target=timeline.target(now)
            if target is None:
                if started is not None:raise RuntimeError('RTC action prefix exhausted')
                arm.joints(previous[:6])
                time.sleep(max(0,1/120-(time.monotonic()-now)))
                previous_time=time.monotonic();continue
            if started is None:
                # A visible target proves a valid new prediction was published.
                # Initialize its heartbeat before enabling watchdog running mode.
                guard.policy_time.value=now
                started=now;guard.running.value=1;previous_time=now
                print('执行中：30Hz动作时间轴，120Hz插值；速度衔接%d帧，输出滤波τ=%.0fms；Space/Esc/Ctrl+C 阻尼停止。'%(
                    config['chunk_smoothing_steps'],1000*config['command_filter_tau_s']),flush=True)
            command=smoother.step(target,max(now-previous_time,1e-6))
            arm.joints(command[:6]);arm.grip(command[6],config['gripper_force_n'])
            previous=command;previous_time=now
            if now>next_report:
                print('t=%.1fs policy RTT=%.0fms'%(now-started,
                      1000*np.median(producer.latencies[-10:]) if producer.latencies else 0),flush=True)
                next_report=now+2
            time.sleep(max(0,1/120-(time.monotonic()-now)))
    except BaseException as error:
        guard.stop.value=1
        control_error=dict(message=str(error),type=type(error).__name__,t=time.monotonic())
        try:control_error.update(
                           measured_joint_deg=np.degrees(state[:6]).tolist(),gripper_m=float(state[6]),
                           feedback_times=np.asarray(stamps).tolist(),
                           last_joint_command_deg=np.degrees(previous[:6]).tolist(),
                           last_gripper_command_m=float(previous[6]))
        except Exception as diagnostic_error:control_error['snapshot_error']=str(diagnostic_error)
        raise
    finally:
        guard.stop.value=1
        producer.thread.join(.1)
        if producer.error:print('Policy producer:',producer.error,flush=True)
        producer.save_report(root,control_error=control_error)
