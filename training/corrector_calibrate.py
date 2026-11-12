"""Calibrate normal-trajectory event rate using the deployed capture-time monitor.

This is validation/calibration on the six existing held-out episodes, not an
independent success-rate evaluation or a substitute for physical trials.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from corrector.monitor import LatentMonitor,RobustGate
from corrector.vendor.vla_corrector.MLP import SiglipResidualMLP
from corrector.vendor.vla_corrector.config import SiglipMLPConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.04)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=SiglipResidualMLP(SiglipMLPConfig(token_dim=payload['model_config']['token_dim'],action_dim=7)).cuda().eval()
    model.load_state_dict(payload['state_dict']);meta=json.loads((args.assets/'metadata.json').read_text())
    settings=dict(meta['monitor_config']);settings['floor']=3.
    def predict(z,a):
        with torch.no_grad():return model(torch.as_tensor(z,device='cuda')[None],torch.as_tensor(a,device='cuda')[None])[0].cpu().numpy()
    episodes=[];elapsed=0;started=time.monotonic()
    for path in sorted(args.features.glob('val_*/complete.json')):
        z=np.load(path.parent/'z.npy',mmap_mode='r');a=np.load(path.parent/'actions.npy',mmap_mode='r')
        monitor=LatentMonitor(predict,payload['contract']['k_step'],settings);scores=[]
        for frame in range(0,len(z),3):
            result=monitor.observe(np.array(z[frame],dtype=np.float32),np.array(a[frame],dtype=np.float32),frame/30.)
            if 'error' in result:scores.append((frame/30.,float(result['error'])))
        episodes.append(dict(episode=path.parent.name,seconds=len(z)/30.,scores=scores));elapsed+=len(z)/30.
        print(json.dumps(dict(episode=path.parent.name,monitor_scores=len(scores),elapsed_s=time.monotonic()-started)),flush=True)
    all_scores=np.array([error for row in episodes for _,error in row['scores']]);candidates=[];chosen=None
    for quantile in (.95,.975,.99,.995,.999,1.):
        floor=float(np.quantile(all_scores,quantile));cfg=dict(meta['monitor_config'],floor=floor);counts=[]
        for row in episodes:
            gate=RobustGate(cfg);counts.append(sum(gate.update(error,stamp)['event'] for stamp,error in row['scores']))
        result=dict(quantile=quantile,floor=floor,events=sum(counts),events_per_minute=sum(counts)/elapsed*60,by_episode=counts)
        candidates.append(result)
        if chosen is None and sum(counts)<=1:chosen=result
    chosen=chosen or candidates[-1]
    meta['monitor_config'].update(floor=chosen['floor'],calibration_monitor_hz=10,
        calibration_normal_events=chosen['events'],calibration_normal_seconds=elapsed,
        calibration_normal_events_per_minute=chosen['events_per_minute'],calibration_quantile=chosen['quantile'])
    report=dict(status='calibrated',physical_robot_tested=False,validation_episodes=len(episodes),
        simulated_monitor_hz=10,action_fps=30,k_step=payload['contract']['k_step'],normal_seconds=elapsed,
        scores=len(all_scores),candidates=candidates,chosen=chosen,elapsed_s=time.monotonic()-started)
    (args.assets/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (args.assets/'normal_calibration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
