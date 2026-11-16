"""Optional pi05 compute backend; execution and arm protocol stay independent."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

BACKENDS=('jax','realtime-vla')
STATES={'policy':('realtime_vla_policy_state.json',8004),
        'corrector':('realtime_vla_corrector_state.json',8005)}


def select(settings,backend=None):
    value=backend or settings.get('pi05',{}).get('inference_backend','jax')
    if value not in BACKENDS:raise ValueError('Unknown pi05 backend: '+str(value))
    out=dict(settings,inference_backend=value)
    if value=='realtime-vla':out.update(policy_port=8004,corrector_policy_port=8005,policy_startup_timeout_s=600)
    return out


def preflight(root,settings,mode='policy',assets=None):
    root=Path(root);step=settings['model_step'];directory=root/'realtime_vla_assets'/('pi05_%d'%step)
    required=[root/'.realtime-vla-venv/bin/python',root/'vendor/realtime_vla/pi05_infer.py',
              directory/'metadata.json',directory/'model.safetensors',root/'logs'/('realtime_vla_validation_%d.json'%step)]
    if mode=='corrector':
        if step!=2500:raise ValueError('Corrector is trained for2500 only')
        required.extend(Path(assets)/n for n in ['metadata.json','norm_stats.json','mlp.npz'])
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError('realtime-vla assets/validation missing: '+', '.join(missing))
    meta=json.loads((directory/'metadata.json').read_text())
    report=json.loads((root/'logs'/('realtime_vla_validation_%d.json'%step)).read_text())
    if (meta['training_updates']!=step or meta['checkpoint_index']!=settings['checkpoint_index']
            or report.get('status')!='passed' or report.get('cases',0)<3):
        raise RuntimeError('realtime-vla checkpoint identity or offline validation failed')
    return directory


def stop_record(root,path,module,port,guard=None):
    from rtc_piper.model_services import process_args,connected
    try:state=json.loads(Path(path).read_text())
    except (OSError,ValueError):return
    args=process_args(state.get('pid'))
    if not args:return
    if (state.get('owner_root')!=str(root) or args!=state.get('argv')
            or module not in args or str(root) not in args):
        raise RuntimeError('Model PID ownership changed; left untouched: '+str(path))
    if connected(port):raise RuntimeError('Model has a connected client; finish that session first')
    if guard:guard.check()
    os.kill(state['pid'],signal.SIGTERM);until=time.monotonic()+10
    while process_args(state['pid']):
        if guard:guard.check()
        if time.monotonic()>until:raise RuntimeError('Idle model did not exit')
        time.sleep(.05)


def stop_idle(root,guard=None,keep=None):
    root=Path(root).resolve()
    for mode,(name,port) in STATES.items():
        if mode!=keep:stop_record(root,root/'logs'/name,'realtime_vla_runtime.serve',port,guard)


def ensure(root,settings,guard=None,mode='policy',assets=None):
    root=Path(root).resolve();preflight(root,settings,mode,assets)
    from rtc_piper.policy import Client
    from rtc_piper.model_services import stop_idle_other,stop_idle_corrector,process_args
    from corrector.client import Client as CorrectorClient,Session
    name,port=STATES[mode];path=root/'logs'/name
    def connect():
        if guard:guard.check()
        client=(Client(settings) if mode=='policy' else CorrectorClient(port,Session('monitor'),backend='realtime-vla'))
        if mode=='corrector':
            expected=json.loads((Path(assets)/'metadata.json').read_text())
            if client.metadata.get('contract')!=expected['contract'] or client.metadata.get('corrector_epoch')!=expected['epoch']:
                client.close();raise RuntimeError('Wrong realtime-vla Corrector assets')
        return client
    try:return connect()
    except (OSError,TimeoutError):pass
    stop_idle(root,guard,keep=mode)
    # Stop only owned idle services, after all requested assets have passed
    # preflight. The caller holds the common follower lock.
    stop_idle_corrector(root,guard)
    stop_idle_other(root,'vpp2',guard);stop_idle_other(root,'pi05',guard)
    stop_record(root,root/'logs/rlt_service.json','rlt_piper.serve',8003,guard)
    try:state=json.loads(path.read_text())
    except (OSError,ValueError):state={}
    argv=process_args(state.get('pid'))
    if argv and (argv!=state.get('argv') or state.get('owner_root')!=str(root)):
        raise RuntimeError('realtime-vla PID changed; left untouched')
    process=None
    if not argv:
        command=[str(root/'.realtime-vla-venv/bin/python'),'-u','-m','realtime_vla_runtime.serve',
                 '--root',str(root),'--step',str(settings['model_step']),'--mode',mode]
        if assets:command.extend(['--assets',str(assets)])
        env=dict(os.environ,PYTHONPATH=':'.join(map(str,[root,root/'vendor/openpi/src',root/'vendor/openpi/packages/openpi-client/src'])),
                 JAX_PLATFORMS='cpu',OPENPI_DATA_HOME=str(root/'cache/openpi'),
                 TRITON_CACHE_DIR=str(root/'cache/realtime_vla_triton'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (root/'logs'/('realtime_vla_'+mode+'.log')).open('a') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('正在准备 realtime-vla Triton 后端（%s），首次编译会较慢。'%mode,flush=True)
    until=time.monotonic()+600
    while time.monotonic()<until:
        if guard:guard.check()
        if process is not None and process.poll() is not None:
            raise RuntimeError('realtime-vla startup failed; see logs/realtime_vla_%s.log'%mode)
        try:return connect()
        except (OSError,TimeoutError):time.sleep(.2)
    raise RuntimeError('realtime-vla startup timed out')
