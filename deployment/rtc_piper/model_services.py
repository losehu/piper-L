"""Start the selected local service; release only this deployment's idle model."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process_args(pid):
    try:return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
    except OSError:return []


def connected(port):
    for path in ['/proc/net/tcp','/proc/net/tcp6']:
        try:rows=Path(path).read_text().splitlines()[1:]
        except OSError:continue
        if any(int(row.split()[1].split(':')[-1],16)==port and row.split()[3]=='01' for row in rows):return True
    return False


def stop_idle_corrector(root,guard=None):
    """Release only an idle, owned Corrector when choosing an ordinary model."""
    root=Path(root).resolve()
    try:state=json.loads((root/'logs/corrector_policy_state.json').read_text())
    except (OSError,ValueError):return
    args=process_args(state.get('pid'))
    if not args:return
    if (state.get('owner_root')!=str(root) or args!=state.get('argv')
            or 'corrector.serve' not in args or str(root) not in args):
        raise RuntimeError('Corrector PID identity changed; left it untouched')
    if connected(state.get('port',8002)):raise RuntimeError('Corrector has a connected client; finish that session first')
    if guard:guard.check()
    os.kill(state['pid'],signal.SIGTERM);until=time.monotonic()+10
    while process_args(state['pid']):
        if guard:guard.check()
        if time.monotonic()>until:raise RuntimeError('Idle Corrector service did not exit')
        time.sleep(.05)


def prepare_pi05(root,settings,guard=None):
    """Caller owns the arm lock; only switch an idle, owned pi05 process."""
    root=Path(root).resolve();checkpoint=root/settings['checkpoint']
    if not (checkpoint/'params').is_dir() or not (checkpoint/'assets').is_dir():
        raise RuntimeError('pi05 checkpoint assets missing: '+str(checkpoint))
    try:state=json.loads((root/'logs/policy_state.json').read_text())
    except (OSError,ValueError):return
    if not process_args(state.get('pid')):return
    metadata=state.get('metadata',{})
    step=metadata.get('training_updates',state.get('training_updates'))
    index=metadata.get('source_checkpoint_index',state.get('source_checkpoint_index'))
    if (step,index)==(settings['model_step'],settings['checkpoint_index']):return
    stop_idle_other(root,'vpp2',guard)


def stop_idle_other(root,selected,guard=None):
    """Caller owns the shared arm lock. Unknown PIDs and other jobs are untouched."""
    root=Path(root).resolve()
    path=root/'logs'/('policy_state.json' if selected=='vpp2' else 'vpp2_service.json')
    try:state=json.loads(path.read_text())
    except (OSError,ValueError):return
    pid=state.get('pid');args=process_args(pid)
    if not args:return
    if selected=='vpp2':
        owned='rtc_piper.serve_policy' in args and str(root) in args
        port=8000
    else:
        owned=(state.get('owner_root')==str(root) and args==state.get('argv')
               and str(root/'vpp2_runtime/server.py') in args)
        port=state.get('port',8766)
    if not owned:raise RuntimeError('Other model PID identity changed; left it untouched')
    if connected(port):raise RuntimeError('Other model has a connected client; finish that session before switching')
    if guard:guard.check()
    print('切换模型：退出本部署的空闲%s服务以释放显存。'%('π0.5' if selected=='vpp2' else 'VPP2'),flush=True)
    os.kill(pid,signal.SIGTERM)
    until=time.monotonic()+10
    while process_args(pid):
        if guard:guard.check()
        if time.monotonic()>until:raise RuntimeError('Previous model service did not exit')
        time.sleep(.05)


def check_vpp2_assets(root,settings):
    if settings.get('execution_mode')=='h100_async_rtc':
        from rtc_piper.vpp2_client import Vpp2Client
        client=Vpp2Client(settings);client.close()
        return None,None
    root=Path(root).resolve();assets=Path(settings['vpp2_root'])
    python=assets/'.venv/bin/python'
    required=[python,assets/'config/train_config.yaml',assets/'weights/best_action.pt',
              assets/'weights/video_int8.safetensors',assets/'weights/video_int8_manifest.json',
              assets/'weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth',
              assets/'weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
              assets/'assets/dataset_stats.json',
              assets/'assets/text_context.pt',root/'offline_sample.npz']
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise RuntimeError('VPP2 assets/environment missing: '+', '.join(missing))
    return assets,python


def start_vpp2(root,settings):
    if settings.get('execution_mode')=='h100_async_rtc':raise RuntimeError('H100 bridge unavailable; local fallback is disabled')
    root=Path(root).resolve();assets,python=check_vpp2_assets(root,settings)
    argv=[str(python),'-u',str(root/'vpp2_runtime/server.py'),'--root',str(assets),
          '--backend','int8','--host','127.0.0.1','--port',str(settings['policy_port']),
          '--warmup-sample',str(root/'offline_sample.npz'),
          '--denoising-steps',str(settings['denoising_steps'])]
    if settings.get('compile_vpp2'):argv.append('--compile-action')
    path=root/'logs/vpp2_service.json'
    try:previous=json.loads(path.read_text())
    except (OSError,ValueError):previous={}
    if previous.get('argv')==argv and process_args(previous.get('pid'))==argv:return None
    env=dict(os.environ,HF_HUB_OFFLINE='1',VPP2_TORCH_LOAD_MMAP='1',OMP_NUM_THREADS='4',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env['PYTHONPATH']=str(root/'vendor/vpp2/src')
    env['TORCHINDUCTOR_CACHE_DIR']=str(root/'cache/vpp2_inductor')
    with (root/'logs/vpp2_policy.log').open('a') as log:
        process=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    path.write_text(json.dumps(dict(pid=process.pid,argv=argv,owner_root=str(root),
        port=settings['policy_port'],asset_root=str(assets),
        compile_vpp2=settings.get('compile_vpp2',False),denoising_steps=settings['denoising_steps']),indent=2)+'\n')
    return process


def stop_changed_vpp2(root,settings,guard=None):
    if settings.get('execution_mode')=='h100_async_rtc':
        return stop_idle_other(root,'pi05',guard)
    try:previous=json.loads((Path(root)/'logs/vpp2_service.json').read_text())
    except (OSError,ValueError):return
    if (bool(previous.get('compile_vpp2',False))!=settings.get('compile_vpp2',False)
            or previous.get('denoising_steps',10)!=settings['denoising_steps']):
        stop_idle_other(root,'pi05',guard)
