"""Independent Corrector entrypoint; leaves all existing model entrypoints intact."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np

ROOT=Path(os.environ.get('PIPER_DEPLOY_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()


def ensure_service(root,assets,settings,guard=None):
    from rtc_piper.model_services import process_args,stop_idle_other
    from corrector.client import Client,Session
    for name in ('mlp.npz','metadata.json','norm_stats.json'):
        if not (assets/name).is_file():raise RuntimeError('Corrector asset missing: '+str(assets/name))
    expected=json.loads((assets/'metadata.json').read_text())
    def connect():
        client=Client(8002,Session('monitor'))
        if (client.metadata.get('contract')!=expected['contract']
                or client.metadata.get('corrector_epoch')!=expected['epoch']):
            client.close();raise RuntimeError('Existing Corrector service has different assets; finish and restart it')
        return client
    try:client=connect();client.close();return
    except (OSError,TimeoutError):pass
    # Caller owns the common arm lock. Only this deployment's idle services
    # are released; active controllers and unknown GPU processes are untouched.
    stop_idle_other(root,'vpp2',guard);stop_idle_other(root,'pi05',guard)
    state_path=root/'logs/corrector_policy_state.json'
    try:state=json.loads(state_path.read_text())
    except (OSError,ValueError):state={}
    args=process_args(state.get('pid'))
    owned=('-m' in args and 'corrector.serve' in args and str(root) in args and state.get('owner_root')==str(root))
    if args and not owned:raise RuntimeError('Corrector PID changed; left untouched')
    if not args:
        with (root/'logs/corrector_policy.log').open('a') as log:
            process=subprocess.Popen(['bash',str(root/'start_corrector.sh'),str(assets)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    else:process=None
    until=time.monotonic()+180
    while time.monotonic()<until:
        if guard:guard.check()
        if process is not None and process.poll() is not None:raise RuntimeError('Corrector startup failed; see logs/corrector_policy.log')
        try:client=connect();client.close();return
        except (OSError,TimeoutError):time.sleep(.2)
    raise RuntimeError('Corrector model startup timed out')


def check(root,settings,assets,mode):
    from rtc_piper.hardware import follower_lock
    from rtc_piper.observations import Observation
    from rtc_piper.policy import validate_actions
    from corrector.client import Client,Session
    with follower_lock(settings):
        ensure_service(root,assets,settings);session=Session(mode);client=Client(8002,session)
        try:
            with np.load(root/'offline_sample.npz',allow_pickle=False) as sample:
                state=np.array(sample['state']);images={k:np.array(sample[k]) for k in ('cam0','cam1')}
            values=[]
            for i in range(5):
                stamp=time.monotonic();obs=Observation(stamp,state,images,np.full(4,stamp))
                before=time.monotonic();result=client.request('infer',obs,state,0,settings['prompt'])
                if result.get('needs_replan'):raise RuntimeError('Unexpected event in static offline check')
                validate_actions(result['actions'],settings);values.append(time.monotonic()-before)
            record=dict(status='passed',mode=mode,hardware_opened=False,robot_commands_sent=0,
                p50_ms=1000*float(np.median(values)),p95_ms=1000*float(np.percentile(values,95)),metadata=client.metadata)
            (root/'logs/corrector_offline_check.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
        finally:client.close()


def execute(root,settings,assets,mode):
    from rtc_piper.hardware import follower_lock,check_interface,SdkArm
    from rtc_piper.observations import CaptureBank,RealSense
    from rtc_piper.safety import Guard,terminal_mode
    from rtc_piper.control import wait_ready,choose_start,prepare
    from rtc_piper.cli import live_settings
    from corrector.client import Client,Session
    from corrector.execution import run
    guard=Guard(settings['channel'],policy_timeout_s=.8);arm=cameras=lock=policy=observer=None
    def signal_stop(*_):
        guard.stop.value=1
        if not guard.armed.value:raise KeyboardInterrupt
    old={sig:signal.signal(sig,signal_stop) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        guard.start();candidate=follower_lock(settings);candidate.__enter__();lock=candidate
        ensure_service(root,assets,settings,guard);session=Session(mode)
        policy=Client(8002,session);observer=Client(8002,session);policy.request('status')
        guard.check();check_interface(settings);settings=live_settings(settings)
        bank=CaptureBank();arm=SdkArm(settings,bank,guard);cameras=RealSense(settings,bank)
        state,stamps,code,enabled,obs=wait_ready(arm,cameras,bank,guard)
        starts=json.loads((root/'training_start_states.json').read_text())
        if code==0 and any(enabled):raise RuntimeError('Arm already enabled by another session')
        row,target=choose_start(state,starts,settings)
        print('π0.5 Corrector mode=%s：自动准备后开始动作；示范起点%d。'%(mode,row),flush=True)
        print('固件恢复可能短暂失能，请提前承托机械臂并清空通路。Space/Esc/Ctrl+C 阻尼停止。',flush=True)
        with terminal_mode(guard):
            after,previous=prepare(arm,cameras,bank,guard,starts,settings)
            run(root,arm,cameras,bank,guard,policy,observer,previous,after,settings)
    finally:
        guard.close()
        if policy:policy.close()
        if observer:observer.close()
        if cameras:cameras.close()
        if arm:arm.close()
        if lock:lock.__exit__(None,None,None)
        for sig,handler in old.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('check','run'),nargs='?',default='check')
    p.add_argument('--mode',choices=('monitor','truncate','full'),default='monitor')
    p.add_argument('--assets',type=Path,default=ROOT/'corrector_assets/pi05_2500_k5')
    p.add_argument('--config',type=Path,default=ROOT/'config.yaml');args=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    settings=select_model(config(args.config),'pi05',ROOT)
    if settings['model_step']!=2500 or settings['checkpoint_index']!=2499:raise ValueError('Corrector is trained for2500 only')
    (ROOT/'logs').mkdir(exist_ok=True)
    os.environ.setdefault('DISPLAY',':0');os.environ.setdefault('XAUTHORITY','/run/user/%d/gdm/Xauthority'%os.getuid())
    if args.command=='check':check(ROOT,settings,args.assets.resolve(),args.mode)
    else:execute(ROOT,settings,args.assets.resolve(),args.mode)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Cancelled; damping stop remains latched.')
    except (RuntimeError,ValueError,OSError) as error:raise SystemExit('停止/未启动：'+str(error))
