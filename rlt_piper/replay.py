"""Actual RLT RPC +120Hz software arm; saved images, synthetic timestamps."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class Finished(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    from rtc_piper.hardware import follower_lock
    from rtc_piper.observations import Observation
    from rlt_piper.cli import start,stop
    from rlt_piper.client import Client
    from rlt_piper.control import run
    from rlt_piper.recording import Recorder
    cfg=select_model(config(a.root/'config.yaml'),'pi05',a.root)
    with np.load(a.root/'offline_sample.npz',allow_pickle=False) as s:
        state=s['state'].copy();images={k:s[k].copy() for k in ['cam0','cam1']}
    initial=state.copy();initial[:6]=np.clip(initial[:6],cfg['policy_command_lower_rad'],cfg['policy_command_upper_rad'])
    with follower_lock(cfg):
        process=start(a.root/'rlt_assets/pi05_2500','collect');policy=None
        try:
            policy=Client('collect');epoch=time.monotonic();end=epoch+a.seconds
            class Guard:
                stop=SimpleNamespace(value=0);running=SimpleNamespace(value=0)
                policy_time=SimpleNamespace(value=epoch);startup_deadline=SimpleNamespace(value=end)
                def check(self):
                    if time.monotonic()>=end:raise Finished()
                    if self.stop.value:raise RuntimeError('Software guard stopped')
            class Cameras:
                def check(self):pass
            class Bank:
                def observation(self):
                    t=epoch+np.floor((time.monotonic()-epoch)*30)/30-1/30
                    return Observation(t,state.copy(),images,np.full(4,t))
                def live(self):return state.copy(),np.full(4,time.monotonic())
            class Arm:
                def status(self,**kw):pass
                def joints(self,q):
                    if np.shape(q)!=(6,) or not np.isfinite(q).all():raise RuntimeError('Invalid software target')
                def grip(self,q,force):
                    if not 0<=q<=.1:raise RuntimeError('Invalid software gripper')
            recorder=Recorder(a.output.parent/'software_only','collect',policy.metadata['rlt_token_sha256'],cfg)
            report=dict(status='failed',hardware_opened=False,robot_commands_sent=0,corrector_enabled=False,
                        source='repeated saved images/state; synthetic replay timestamps, no formal learning data')
            try:run(a.root,Arm(),Cameras(),Bank(),Guard(),policy,initial,epoch-.2,cfg,recorder)
            except Finished:report['status']='passed'
            finally:
                gaps=np.diff([t for t,q in recorder.commands]);ages=[r['received']-r['t'] for r in recorder.plans]
                report.update(accepted=len(recorder.plans),software_commands=len(recorder.commands),
                    max_control_gap_ms=float(1000*max(gaps)) if len(gaps) else None,
                    capture_to_action_p50_ms=float(1000*np.median(ages)) if ages else None)
                a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
        finally:
            if policy:policy.close()
            stop(process)

if __name__=='__main__':main()
