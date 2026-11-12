#!/usr/bin/env python3
"""Start/reuse the model-only H100 service and keep private SSH forwards alive.

This command never enables or moves the robot. The robot has a separate CLI.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'deployment/h100_runtime.json')
    parser.add_argument('--stop',action='store_true');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());logs=args.config.resolve().parent/'logs';logs.mkdir(exist_ok=True)
    state_path=logs/'h100_bridge_state.json'
    if args.stop:
        state=json.loads(state_path.read_text());pid=state['pid']
        try:cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        except OSError:cmd=b''
        if cmd:
            if str(Path(__file__).resolve()).encode() not in cmd:raise RuntimeError('Bridge PID identity changed; untouched')
            os.kill(pid,signal.SIGTERM)
        print('Bridge stop requested; H100 model remains warm.');return
    if state_path.exists():
        try:
            old=json.loads(state_path.read_text());cmd=(Path('/proc')/str(old['pid'])/'cmdline').read_bytes()
        except (OSError,ValueError,KeyError):cmd=b''
        if cmd and str(Path(__file__).resolve()).encode() in cmd:
            raise RuntimeError('An owned bridge is already active; use it or --stop before restarting')
    remote_code='CFG='+repr(cfg)+'''
from pathlib import Path
import json,os,socket,subprocess
r=Path(CFG['remote_deployment']);logs=r/'logs';logs.mkdir(exist_ok=True)
argv=[CFG['remote_python'],'-u','-m','vpp2_network.server','--assets',CFG['remote_assets'],
      '--sample',CFG['remote_sample'],'--port',str(CFG.get('remote_port',19868))]
if CFG.get('cuda_graphs',True):argv.append('--cuda-graphs')
if CFG.get('compile_prefill',True):argv.append('--compile-prefill')
p=logs/'network_server_process.json'
old=json.loads(p.read_text()) if p.exists() else {}
try:current=[s.decode() for s in (Path('/proc')/str(old.get('pid'))/'cmdline').read_bytes().split(b'\\0') if s]
except OSError:current=[]
if current:
    if current!=old.get('argv') or current!=argv:raise RuntimeError('Remote model PID/config differs; left untouched')
    print(json.dumps({'reused':True,'pid':old['pid']}))
else:
    apps=subprocess.check_output(['nvidia-smi','-i',str(CFG.get('gpu',0)),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('Selected H100 has compute jobs; left untouched')
    with socket.socket() as s:s.bind(('127.0.0.1',CFG.get('remote_port',19868)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(CFG.get('gpu',0)),OMP_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',VPP2_TORCH_LOAD_MMAP='1',
        PYTHONPATH=str(r)+':'+CFG['remote_vendor_src'],
        TORCHINDUCTOR_CACHE_DIR=CFG.get('remote_cache',str(r/'cache/inductor')),
        VPP2_SERVER_STATE=str(logs/'network_server_state.json'))
    with (logs/'network_server.log').open('a') as f:
        process=subprocess.Popen(argv,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    p.write_text(json.dumps({'pid':process.pid,'argv':argv,'owner_root':str(r)},indent=2)+'\\n')
    print(json.dumps({'started':True,'pid':process.pid}))
'''
    started=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',cfg['h100_ssh'],'python3 -'],
        input=remote_code,text=True,capture_output=True)
    if started.returncode:raise RuntimeError(started.stderr[-3000:])
    print('H100模型服务已准备，正在建立连接。',flush=True)
    local_port=int(cfg.get('local_forward_port',21968));robot_port=int(cfg.get('robot_forward_port',21969));remote_port=int(cfg.get('remote_port',19868))
    for p in [local_port,robot_port,remote_port]:
        if not 1024<=p<=65535:raise ValueError('Invalid forward port')
    state=dict(pid=os.getpid(),config=str(args.config.resolve()),status='connecting',started=time.time())
    state_path.write_text(json.dumps(state,indent=2)+'\n');processes=[]
    def terminate(*_):raise KeyboardInterrupt
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    common=['ssh','-N','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ControlPath=none',
        '-o','Compression=no','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
    try:
        with (logs/'h100_bridge_ssh.log').open('a') as errors:
            for argv in [common+['-L','127.0.0.1:%d:127.0.0.1:%d'%(local_port,remote_port),cfg['h100_ssh']],
                         common+['-R','127.0.0.1:%d:127.0.0.1:%d'%(robot_port,local_port),cfg['robot_ssh']]]:
                processes.append(subprocess.Popen(argv,stdout=subprocess.DEVNULL,stderr=errors))
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));until=time.monotonic()+600
            while time.monotonic()<until:
                if any(p.poll() is not None for p in processes):raise RuntimeError('SSH forward exited; see h100_bridge_ssh.log')
                try:
                    with opener.open('http://127.0.0.1:%d/health'%local_port,timeout=3) as response:health=json.load(response)
                    if (health.get('checkpoint_step')!=2250 or health.get('denoising_steps')!=5
                        or not health.get('compiled_action') or health.get('backend')!='bf16'
                        or health.get('robot_connected') is not False):raise RuntimeError('Wrong H100 model identity')
                    if health.get('warmed_up'):break
                except (OSError,http.client.HTTPException):pass
                time.sleep(3)
            else:raise RuntimeError('H100 model warmup timed out')
            check="import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:%d/health',timeout=5); print(json.load(r)['checkpoint_step'])"%robot_port
            verified=subprocess.run(['ssh','-o','BatchMode=yes',cfg['robot_ssh'],'python3 -'],input=check,text=True,capture_output=True,timeout=15)
            if verified.returncode or verified.stdout.strip()!='2250':raise RuntimeError('Collector-to-H100 forward check failed')
            state.update(status='ready',ready=time.time(),ssh_pids=[p.pid for p in processes],health=health)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            print('H100单卡连接已就绪；未启动机械臂。保持此进程运行，Ctrl+C关闭连接。',flush=True)
            while True:
                if any(p.poll() is not None for p in processes):raise RuntimeError('SSH forward lost; bridge stopped')
                time.sleep(1)
    except KeyboardInterrupt:pass
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        state.update(status='stopped',stopped=time.time());state_path.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as error:raise SystemExit(str(error))
