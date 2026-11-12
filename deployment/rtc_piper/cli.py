"""No-UI entrypoints: run performs automatic preflight, preparation and rollout."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def config(path):
    import yaml
    data=yaml.safe_load(Path(path).read_text())
    if (data['robot']!='piper_l' or data['firmware_driver']!='v189'
            or data['channel']!='can1' or data['fps']!=30 or data['control_hz']!=120
            or not 2<=data['execute_horizon']<=24):
        raise ValueError('Model/robot/rate configuration does not match this deployment')
    for key in ['policy_joint_speed_deg_s']:
        values=np.asarray(data[key],dtype=float)
        if values.shape!=(6,) or not np.isfinite(values).all():raise ValueError('Invalid '+key)
    if not 0<data['max_observation_age_s']<=.25:raise ValueError('Observation expiry may not exceed .25s')
    if not np.all((np.asarray(data['policy_joint_speed_deg_s'])>0)
                  & (np.asarray(data['policy_joint_speed_deg_s'])<=90)):
        raise ValueError('Policy joint speed must be in (0,90] deg/s')
    if (not 0<data['gripper_speed_m_s']<=.2 or not 0<data['gripper_force_n']<=3
            or data['chunk_smoothing_method']!='cubic_velocity_bridge'
            or not .01<=data['command_filter_tau_s']<=.2
            or not 2<=data['chunk_smoothing_steps']<=data['execute_horizon']):
        raise ValueError('Invalid motion/RTC limits')
    if (data['model_step']!=2500 or data['checkpoint_index']!=2499
            or data['checkpoint']!='checkpoints/pi05_piper_2500_bf16'):
        raise ValueError('This deployment must select the requested final2500-update checkpoint')
    from rtc_piper.hardware import apply_hardware_limits
    return apply_hardware_limits(data,data['hardware_limits_cache'])


def live_settings(settings):
    from rtc_piper.hardware import read_hardware_limits,apply_hardware_limits
    readback=read_hardware_limits(settings)
    (ROOT/'logs/firmware_limits_live.json').write_text(json.dumps(readback,indent=2)+'\n')
    return apply_hardware_limits(settings,readback)


def ensure_policy(settings,guard=None,owns_hardware_lock=False):
    from rtc_piper.policy import Client
    from rtc_piper.model_services import stop_idle_other,start_vpp2,check_vpp2_assets,stop_changed_vpp2,stop_idle_corrector
    from rtc_piper.vpp2_client import Vpp2Client
    if guard:guard.check()
    if not owns_hardware_lock:
        from rtc_piper.hardware import follower_lock
        with follower_lock(settings):return ensure_policy(settings,guard,owns_hardware_lock=True)
    kind=settings.get('model_name','pi05')
    if kind=='vpp2':
        check_vpp2_assets(ROOT,settings)
        stop_changed_vpp2(ROOT,settings,guard)
    else:
        from rtc_piper.model_services import prepare_pi05
        prepare_pi05(ROOT,settings,guard)
    stop_idle_corrector(ROOT,guard)
    stop_idle_other(ROOT,kind,guard)
    def connect():
        if guard:guard.check()
        client=Vpp2Client(settings) if kind=='vpp2' else Client(settings)
        try:
            if guard:guard.check()
        except BaseException:
            client.close();raise
        return client
    try:return connect()
    except (OSError,TimeoutError):pass
    if kind=='vpp2':
        process=start_vpp2(ROOT,settings)
        print('正在本机加载VPP2 INT8，日志：',ROOT/'logs/vpp2_policy.log',flush=True)
        until=time.monotonic()+settings['policy_startup_timeout_s']
        while time.monotonic()<until:
            if guard:guard.check()
            if process is not None and process.poll() is not None:
                raise RuntimeError('VPP2 startup failed; inspect logs/vpp2_policy.log')
            time.sleep(.5)
            try:return connect()
            except (OSError,TimeoutError):pass
        raise RuntimeError('VPP2 startup timeout; inspect logs/vpp2_policy.log')
    state_path=ROOT/'logs/policy_state.json'
    state={}
    if state_path.exists():
        try:state=json.loads(state_path.read_text())
        except ValueError:pass
    alive=False
    if state.get('pid'):
        try:
            cmd=Path('/proc/%d/cmdline'%state['pid']).read_bytes()
            alive=b'rtc_piper.serve_policy' in cmd and str(ROOT).encode() in cmd
        except OSError:pass
    process=None
    if not alive:
        log=(ROOT/'logs/policy.log').open('a')
        process=subprocess.Popen(['bash',str(ROOT/'start_policy.sh'),'--step',str(settings['model_step'])],cwd=str(ROOT),
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        log.close()
        print('正在本机加载π0.5模型，日志：',ROOT/'logs/policy.log',flush=True)
    until=time.monotonic()+180
    while time.monotonic()<until:
        if guard:guard.check()
        if process is not None and process.poll() is not None:
            raise RuntimeError('Policy startup failed; inspect logs/policy.log')
        time.sleep(.5)
        try:return connect()
        except (OSError,TimeoutError):pass
    raise RuntimeError('Policy startup timeout; inspect logs/policy.log')


def offline_check(settings):
    from rtc_piper.hardware import follower_lock
    with follower_lock(settings):return _offline_check(settings)


def _offline_check(settings):
    from rtc_piper.observations import Observation
    from rtc_piper.policy import validate_actions, Timeline
    client=ensure_policy(settings,owns_hardware_lock=True);durations=[]
    try:
        with np.load(ROOT/'offline_sample.npz',allow_pickle=False) as sample:
            state=np.array(sample['state']);images={k:np.array(sample[k]) for k in ['cam0','cam1']}
        requests=1 if settings.get('model_name')=='vpp2' else 10
        for index in range(requests):
            now=time.monotonic();obs=Observation(now,state,images,np.full(4,now))
            before=time.monotonic();actions=client.infer(obs,settings['prompt']);elapsed=time.monotonic()-before
            actions=validate_actions(actions,settings)
            if settings.get('model_name')!='vpp2' and settings.get('execution_mode')!='sync_chunks':
                timeline=Timeline(ROOT,settings,now)
                timeline.enqueue(actions,now,time.monotonic())
                if timeline.target(time.monotonic()) is None:raise RuntimeError('Empty RTC action prefix')
            durations.append(elapsed)
        result=dict(status='passed',requests=requests,action_shape=[settings.get('action_horizon',50),7],
                    p50_s=float(np.median(durations)),p95_s=float(np.percentile(durations,95)),
                    metadata=client.metadata,hardware_opened=False)
        (ROOT/'logs/offline_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:client.close()


def sensors(settings,seconds):
    from rtc_piper.hardware import follower_lock,check_interface,SdkArm
    from rtc_piper.observations import CaptureBank,RealSense,NotReady
    from rtc_piper.safety import Guard
    guard=Guard(settings['channel']);arm=cameras=lock=None
    try:
        guard.start()
        candidate=follower_lock(settings);candidate.__enter__();lock=candidate
        check_interface(settings);settings=live_settings(settings)
        bank=CaptureBank();arm=SdkArm(settings,bank,guard)
        cameras=RealSense(settings,bank);samples=[];until=time.monotonic()+seconds;last_error=''
        while time.monotonic()<until:
            guard.check();cameras.check()
            try:
                obs=bank.observation();state,stamps=bank.live();code,enabled=arm.status(allow_estop=True)
                samples.append(float(np.max(np.abs(obs.state_times-obs.t))))
                last_error=''
            except NotReady as error:last_error=str(error)
            time.sleep(.01)
        if len(samples)<20:raise RuntimeError('Sensor check failed: '+last_error)
        result=dict(status='passed',samples=len(samples),max_state_image_error_ms=1000*max(samples),
                    arm_status=code,enabled=enabled,joint_deg=np.degrees(state[:6]).tolist(),
                    sdk_driver=type(arm.arm).__module__,can_control_frames_sent=0,read_only_limit_queries_sent=6,
                    camera_stats=cameras.diagnostics(),capture_diagnostics=bank.diagnostics())
        (ROOT/'logs/sensor_check.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        guard.close()
        if cameras:
            try:
                (ROOT/'logs/camera_health_latest.json').write_text(json.dumps(
                    dict(time=time.time(),cameras=cameras.diagnostics()),indent=2)+'\n')
            except OSError as error:print('相机诊断保存失败：',error,flush=True)
            finally:cameras.close()
        if arm:arm.close()
        if lock:lock.__exit__(None,None,None)


def execute(settings):
    from rtc_piper.hardware import follower_lock,check_interface,SdkArm
    from rtc_piper.observations import CaptureBank,RealSense
    from rtc_piper.policy import Client
    from rtc_piper.safety import Guard,terminal_mode
    from rtc_piper.control import wait_ready,choose_start,prepare,run_policy
    guard=Guard(settings['channel'],policy_timeout_s=settings.get('policy_timeout_s',.8));arm=cameras=lock=client=None
    def signal_stop(*_):
        guard.stop.value=1
        if not guard.armed.value:raise KeyboardInterrupt
    old={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        guard.start()
        print('自动执行模式：预检通过即开始动作；Space/Esc/Ctrl+C 可取消或阻尼停止。',flush=True)
        candidate=follower_lock(settings);candidate.__enter__();lock=candidate
        client=ensure_policy(settings,guard=guard,owns_hardware_lock=True)
        guard.check()
        check_interface(settings);settings=live_settings(settings)
        bank=CaptureBank();arm=SdkArm(settings,bank,guard)
        cameras=RealSense(settings,bank)
        state,stamps,code,enabled,obs=wait_ready(arm,cameras,bank,guard)
        starts=json.loads((ROOT/'training_start_states.json').read_text())
        row,target=choose_start(state,starts,settings)
        if code==0 and any(enabled):raise RuntimeError('Arm already enabled by another session')
        print('当前六轴角度：',np.round(np.degrees(state[:6]),2),flush=True)
        print('固件关节范围(°)：',np.round(np.degrees(np.array([
            settings['policy_command_lower_rad'],settings['policy_command_upper_rad']])).T,1).tolist(),flush=True)
        print('示范准备角度：',np.round(np.degrees(target[:6]),2),flush=True)
        print('即将自动恢复/对齐/执行。固件恢复可能失能下落，请提前承托并清空运动通路。',flush=True)
        print('运行中全局 Space/Esc 或终端 Ctrl+C：固件阻尼停止。',flush=True)
        guard.check()
        with terminal_mode(guard):
            after,previous=prepare(arm,cameras,bank,guard,starts,settings)
            run_policy(ROOT,arm,cameras,bank,guard,client,previous,after,settings)
    finally:
        guard.close()
        if cameras:
            try:
                (ROOT/'logs/camera_health_latest.json').write_text(json.dumps(
                    dict(time=time.time(),cameras=cameras.diagnostics()),indent=2)+'\n')
            except OSError as error:print('相机诊断保存失败：',error,flush=True)
            finally:cameras.close()
        if arm:arm.close()
        if lock:lock.__exit__(None,None,None)
        if client:client.close()
        for sig,handler in old.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description='Piper-L / pi05 or vpp2 local command-line deployment')
    parser.add_argument('command',choices=['check','sensors','run'],nargs='?',default='check')
    parser.add_argument('--config',default=str(ROOT/'config.yaml'))
    parser.add_argument('--seconds',type=float,default=6)
    parser.add_argument('--model',type=str.lower,choices=['pi05','vpp2'],default='pi05')
    parser.add_argument('--compile',dest='compile_vpp2',action='store_const',const=True,default=None)
    parser.add_argument('--pi05-step',type=int,choices=[1500,2500],default=None)
    parser.add_argument('--no-smoothing',action='store_true',help='Disable chunk bridge and output filter, retain command limits')
    parser.add_argument('--no-rtc',action='store_true',help='Execute all50 pi05 frames, then observe and infer; no prefetch')
    args=parser.parse_args();(ROOT/'logs').mkdir(exist_ok=True)
    os.environ.setdefault('DISPLAY',':0')
    os.environ.setdefault('XAUTHORITY','/run/user/%d/gdm/Xauthority'%os.getuid())
    from rtc_piper.profiles import select_model
    settings=select_model(config(args.config),args.model,ROOT,compile_vpp2=args.compile_vpp2,
                          pi05_step=args.pi05_step,no_smoothing=args.no_smoothing,no_rtc=args.no_rtc)
    print('模型：%s，checkpoint=%d，%d帧，调度=%s。'%(
        settings['model_name'],settings['model_step'],settings['action_horizon'],settings['execution_mode']),flush=True)
    if settings['model_name']=='vpp2':
        print('VPP2采样：%d步，动作去噪torch.compile=%s。'%(
            settings['denoising_steps'],settings['compile_vpp2']),flush=True)
    else:print('π0.5平滑：%s。'%('开启' if settings.get('smoothing_enabled',True) else '关闭（保留限位/限速）'),flush=True)
    if args.command=='check':offline_check(settings)
    elif args.command=='sensors':sensors(settings,args.seconds)
    else:execute(settings)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nCancelled; stop remains latched.')
    except (RuntimeError,ValueError,OSError) as error:
        print('\n停止/未启动：'+str(error),flush=True)
        raise SystemExit(1)
