"""Resume-safe eight-GPU pipeline; coexists with other GPU processes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--openpi-root',type=Path,required=True);p.add_argument('--k-step',type=int,default=5)
    args=p.parse_args();root=args.run_root.resolve();source=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True,exist_ok=True);logs=root/'logs';logs.mkdir(exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='preparing',world_size=8,k_step=args.k_step,robot_started=False)
    def status(value,**extra):
        state.update(status=value,time=time.time(),**extra);(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
    env=dict(os.environ,PYTHONPATH=str(source)+':'+str(args.openpi_root/'src')+':'+str(args.openpi_root/'packages/openpi-client/src'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false')
    assets=root/'assets';assets.mkdir(exist_ok=True)
    norm=args.checkpoint/'assets/local/piper_grasp_train/norm_stats.json'
    if not (assets/'norm_stats.json').exists():shutil.copy2(norm,assets/'norm_stats.json')
    if not (assets/'vision.npz').exists():
        status('exporting_encoder')
        subprocess.run([sys.executable,'-u','-m','corrector.vision','--checkpoint',str(args.checkpoint),'--output',str(assets/'vision.npz')],
            cwd=source,env=dict(env,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'),check=True)
    status('extracting_8_gpu');cpus=sorted(os.sched_getaffinity(0));processes=[]
    try:
        for gpu in ([] if len(list((root/'features').glob('*/complete.json')))==54 else range(8)):
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
                XLA_PYTHON_CLIENT_MEM_FRACTION='.08',JAX_COMPILATION_CACHE_DIR=str(root/'jax_cache'))
            argv=[sys.executable,'-u',str(source/'training/corrector_extract.py'),'--data-root',str(args.data_root),
                '--encoder',str(assets/'vision.npz'),'--norm',str(assets/'norm_stats.json'),'--output',str(root/'features'),
                '--rank',str(gpu),'--world-size','8','--batch','4','--k-step',str(args.k_step)]
            selected=cpus[gpu*8:gpu*8+8]
            with (logs/('extract_rank%d.log'%gpu)).open('a') as log:
                process=subprocess.Popen(['taskset','-c',','.join(map(str,selected))]+argv,cwd=source,env=worker_env,stdout=log,stderr=subprocess.STDOUT)
            processes.append(process)
        (logs/'pipeline_extract_processes.json').write_text(json.dumps([p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            failures=[p.returncode for p in processes if p.poll() not in (None,0)]
            if failures:raise RuntimeError('Feature worker failed; inspect logs/extract_rank*.log')
            time.sleep(3)
        if any(p.returncode for p in processes):raise RuntimeError('Feature extraction failed')
        complete=list((root/'features').glob('*/complete.json'))
        if len(complete)!=54:raise RuntimeError('Expected all54 complete episodes')
        status('training_8_gpu',feature_episodes=len(complete))
        train_env=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',NCCL_ASYNC_ERROR_HANDLING='1')
        argv=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes','1','--nproc_per_node','8',
            str(source/'training/corrector_train.py'),'--features',str(root/'features'),'--output',str(root/('run_k%d'%args.k_step)),
            '--k-step',str(args.k_step),'--epochs','30','--microbatch','8','--global-batch','512','--memory-fraction','.10']
        with (logs/'train.log').open('a') as log:subprocess.run(argv,cwd=source,env=train_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('trained',checkpoint=str(root/('run_k%d/best.pt'%args.k_step)))
        subprocess.run([sys.executable,'-u',str(source/'training/corrector_finish.py'),
            '--run-root',str(root),'--k-step',str(args.k_step)],cwd=source,env=env,check=True)
        status('ready_for_deployment',assets=str(root/'deploy'))
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=8)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
