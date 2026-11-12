"""Wait for the owned eight-rank run, then export and calibrate its best model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--k-step',type=int,default=5)
    args=p.parse_args();root=args.run_root.resolve();source=Path(__file__).resolve().parents[1]
    run=root/('run_k%d'%args.k_step);begin=time.monotonic()
    while not (run/'complete.json').exists():
        status=json.loads((root/'status.json').read_text())
        if status['status']=='failed':raise RuntimeError('Training failed: '+status.get('error','unknown'))
        if time.monotonic()-begin>7200:raise RuntimeError('Training completion wait exceeded2h')
        time.sleep(3)
    env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([sys.executable,'-u',str(source/'training/corrector_export.py'),'--checkpoint',str(run/'best.pt'),
        '--norm',str(root/'assets/norm_stats.json'),'--output',str(root/'deploy')],cwd=source,
        env=dict(env,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu'),check=True)
    subprocess.run([sys.executable,'-u',str(source/'training/corrector_calibrate.py'),'--checkpoint',str(run/'best.pt'),
        '--features',str(root/'features'),'--assets',str(root/'deploy')],cwd=source,
        env=dict(env,CUDA_VISIBLE_DEVICES='2'),check=True)
    (root/'export_complete.json').write_text(json.dumps(dict(status='exported_and_calibrated',
        assets=str(root/'deploy'),physical_robot_tested=False),indent=2)+'\n')


if __name__=='__main__':main()
