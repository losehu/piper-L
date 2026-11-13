"""Native RLT check/collect/eval/online entry. No Corrector inference."""
import argparse,json,os,signal,socket,subprocess,time
from pathlib import Path
import numpy as np
from rlt_piper.contract import assets
from rlt_piper.client import Client

ROOT=Path(os.environ.get('PIPER_DEPLOY_ROOT',str(Path(__file__).resolve().parents[1]))).resolve()

def environment():
    return dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'vendor/openpi/src')+':'+str(ROOT/'vendor/openpi/packages/openpi-client/src'),
                OPENPI_DATA_HOME=str(ROOT/'cache/openpi'),JAX_COMPILATION_CACHE_DIR=str(ROOT/'cache/rlt_jax'),
                XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='.65',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')

def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(15)
        except subprocess.TimeoutExpired:raise RuntimeError('RLT model did not exit; no other process was killed')
    if process is not None:
        path=ROOT/'logs/rlt_service.json'
        try:
            state=json.loads(path.read_text())
            if state.get('pid')==process.pid:
                state.update(status='stopped',stopped_at=time.time());temp=path.with_suffix('.json.partial')
                temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)
        except (OSError,ValueError):pass

def start(directory,stage,guard=None):
    assets(directory,stage!='collect')
    from rtc_piper.model_services import stop_idle_corrector,stop_idle_other
    stop_idle_corrector(ROOT,guard);stop_idle_other(ROOT,'vpp2',guard);stop_idle_other(ROOT,'pi05',guard)
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8003))==0:raise RuntimeError('RLT port8003 is already occupied; no unknown service stopped')
    argv=[str(ROOT/'vendor/openpi/.venv/bin/python'),'-u','-m','rlt_piper.serve','--root',str(ROOT),'--assets',str(directory),'--stage',stage]
    with (ROOT/'logs/rlt_policy.log').open('a') as log:
        process=subprocess.Popen(argv,cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        until=time.monotonic()+180
        while time.monotonic()<until:
            if guard:guard.check()
            if process.poll() is not None:raise RuntimeError('RLT service startup failed; see logs/rlt_policy.log')
            try:c=Client(stage);c.close();return process
            except (OSError,TimeoutError):time.sleep(.2)
        raise RuntimeError('RLT model startup timed out')
    except BaseException:stop(process);raise

def train(directory):
    subprocess.run([str(ROOT/'vendor/openpi/.venv/bin/python'),'-u','-m','rlt_piper.learn','--root',str(ROOT),'--assets',str(directory)],
                   cwd=ROOT,env=environment(),check=True)

def check(directory,stage,settings):
    from rtc_piper.hardware import follower_lock
    from rtc_piper.observations import Observation
    from rtc_piper.policy import validate_actions
    with follower_lock(settings):
        process=start(directory,stage);client=None
        try:
            client=Client(stage)
            with np.load(ROOT/'offline_sample.npz',allow_pickle=False) as s:state=s['state'];images={k:s[k] for k in ['cam0','cam1']}
            timings=[]
            for _ in range(10):
                t=time.monotonic();obs=Observation(t,state,images,np.full(4,t))
                result=client.request('infer',obs,prompt=settings['prompt']);validate_actions(result['actions'],settings)
                if result['rlt']['z_rl'].shape!=(512,):raise RuntimeError('Unexpected RLT feature shape')
                timings.append(time.monotonic()-t)
            record=dict(status='passed',hardware_opened=False,robot_commands_sent=0,stage=stage,corrector_enabled=False,
                        p50_ms=float(np.median(timings)*1000),p95_ms=float(np.percentile(timings,95)*1000),metadata=client.metadata)
            (ROOT/'logs/rlt_offline_check.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
        finally:
            if client:client.close()
            stop(process)

def execute(directory,stage,settings,learn):
    from rtc_piper.hardware import follower_lock,check_interface,SdkArm
    from rtc_piper.observations import CaptureBank,RealSense
    from rtc_piper.safety import Guard,terminal_mode
    from rtc_piper.control import wait_ready,choose_start,prepare
    from rtc_piper.cli import live_settings
    from rlt_piper.control import run
    from rlt_piper.recording import Recorder
    from rlt_piper.labels import Labels
    guard=Guard(settings['channel'],policy_timeout_s=.8);process=policy=arm=cameras=labels=recorder=lock=None
    terminal=None;error=None
    def signal_stop(*_):
        guard.stop.value=1
        if not guard.armed.value:raise KeyboardInterrupt
    old={sig:signal.signal(sig,signal_stop) for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]}
    try:
        candidate=follower_lock(settings);candidate.__enter__();lock=candidate
        guard.start();process=start(directory,stage,guard);policy=Client(stage)
        guard.check();check_interface(settings);settings=live_settings(settings)
        bank=CaptureBank();arm=SdkArm(settings,bank,guard);cameras=RealSense(settings,bank)
        state,stamps,code,enabled,obs=wait_ready(arm,cameras,bank,guard)
        if code==0 and any(enabled):raise RuntimeError('Arm already enabled by another session')
        starts=json.loads((ROOT/'training_start_states.json').read_text());row,target=choose_start(state,starts,settings)
        print('RLT stage=%s，Corrector关闭；自动准备起点%d。F8成功/F9失败结束并阻尼停止；Space/Esc/Ctrl+C急停。'%(stage,row),flush=True)
        print('固件恢复可能短暂失能，请承托机械臂并清空对齐通路。',flush=True)
        recorder=Recorder(ROOT,stage,policy.metadata['rlt_token_sha256'],settings)
        with terminal_mode(guard):
            after,previous=prepare(arm,cameras,bank,guard,starts,settings)
            labels=Labels(recorder,guard);labels.start()
            try:run(ROOT,arm,cameras,bank,guard,policy,previous,after,settings,recorder)
            except RuntimeError as e:
                if (recorder.label is None or recorder.error or str(e)!='Emergency stop latched or independent guard unavailable'):
                    recorder.label=None;error=repr(e);raise
            finally:
                labels.close()
                if recorder.label is not None:
                    try:
                        time.sleep(.12);obs=bank.observation();terminal_client=Client(stage)
                        try:result=terminal_client.request('features',obs,prompt=settings['prompt'])
                        finally:terminal_client.close()
                        terminal=dict(t=obs.t,**{k:result['rlt'][k] for k in ['z_rl','proprio','ref_chunk']})
                    except Exception as e:error=str(e)
    finally:
        try:guard.close()
        finally:
            if labels:labels.close()
            if policy:policy.close()
            if cameras:cameras.close()
            if arm:arm.close()
            stop(process)
            if recorder:print('RLT episode:',recorder.save(terminal,error),flush=True)
            if lock:lock.__exit__(None,None,None)
            for sig,handler in old.items():signal.signal(sig,handler)
    if learn and recorder and recorder.label is not None and terminal:train(directory)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','run','train'],default='check',nargs='?')
    p.add_argument('--stage',choices=['collect','eval','online'],default='collect')
    p.add_argument('--assets',type=Path,default=ROOT/'rlt_assets/pi05_2500');p.add_argument('--no-train',action='store_true');a=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    settings=select_model(config(ROOT/'config.yaml'),'pi05',ROOT);directory=a.assets.resolve();assets(directory,a.stage!='collect')
    if a.stage!='collect':
        trained=json.loads((directory/'actor.json').read_text())['execution_settings']
        if any(settings.get(k)!=v for k,v in trained.items()):raise RuntimeError('RLT actor RTC/filter settings do not match this execution')
    (ROOT/'logs').mkdir(exist_ok=True)
    if a.command=='train':
        from rtc_piper.hardware import follower_lock
        with follower_lock(settings):train(directory)
    elif a.command=='check':check(directory,a.stage,settings)
    else:execute(directory,a.stage,settings,not a.no_train and a.stage!='eval')

if __name__=='__main__':main()
