"""Independent frozen-pi05 RLT Token preparation; no robot operation."""
import argparse,json,os,subprocess,sys,shutil,fcntl,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--openpi-root',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);a=p.parse_args();source=Path(__file__).resolve().parents[1]
    a.run_root.mkdir(parents=True,exist_ok=True);logs=a.run_root/'logs';logs.mkdir(exist_ok=True)
    lock=(a.run_root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=a.run_root/'deploy';out.mkdir(exist_ok=True)
    shutil.copy2(a.checkpoint/'assets/local/piper_grasp_train/norm_stats.json',out/'norm_stats.json')
    env=dict(os.environ,PYTHONPATH=str(source)+':'+str(source/'deployment')+':'+str(a.openpi_root/'src')+':'+str(a.openpi_root/'packages/openpi-client/src'),
             OPENPI_DATA_HOME='/data/nvme0/lingfang/home-cache/openpi',XLA_PYTHON_CLIENT_PREALLOCATE='false',
             XLA_PYTHON_CLIENT_MEM_FRACTION='.18',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    def status(s,**kw):
        d=dict(status=s,pid=os.getpid(),robot_started=False,time=time.time(),**kw)
        (a.run_root/'status.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
    workers=[]
    try:
        status('extracting_8_gpu')
        cpus=sorted(os.sched_getaffinity(0))
        for rank in range(8):
            log=(logs/('extract_%d.log'%rank)).open('a')
            argv=[sys.executable,'-u',str(source/'training/rlt_extract.py'),'--data-root',str(a.data_root),'--checkpoint',str(a.checkpoint),
                  '--output',str(a.run_root/'features'),'--rank',str(rank)]
            selected=cpus[rank*8:rank*8+8]
            if not selected:raise RuntimeError('Expected at least64 logical CPUs for eight workers')
            workers.append(subprocess.Popen(['taskset','-c',','.join(map(str,selected))]+argv,env=dict(env,CUDA_VISIBLE_DEVICES=str(rank)),cwd=source,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('Feature worker failed; inspect extract logs')
            time.sleep(2)
        if any(p.returncode for p in workers):raise RuntimeError('Feature extraction failed')
        if len(list((a.run_root/'features').glob('*/complete.json')))!=54:raise RuntimeError('Expected original48/6 episode split')
        status('training_token_8_gpu')
        with (logs/'token.log').open('a') as log:
            subprocess.run([sys.executable,'-u',str(source/'training/rlt_token.py'),'--features',str(a.run_root/'features'),
                '--output',str(out),'--steps',str(a.steps)],env=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('token_ready_actor_requires_labelled_rollouts',assets=str(out))
    except BaseException as e:status('failed',error=repr(e));raise
    finally:
        for p in workers:
            if p.poll() is None:p.terminate()

if __name__=='__main__':main()
