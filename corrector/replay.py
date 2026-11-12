"""Real policy/monitor RPC +120Hz software replay, never robot/CAN devices."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from corrector.client import Client,Session
from corrector.execution import run


class Finished(Exception):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--port',type=int,default=8002);p.add_argument('--mode',choices=('monitor','truncate','full'),default='monitor')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    settings=select_model(config(args.config),'pi05',args.root)
    with np.load(args.sample,allow_pickle=False) as data:
        measured=np.array(data['state']);images={k:np.array(data[k]) for k in ('cam0','cam1')}
    initial=measured.copy();initial[:6]=np.clip(initial[:6],settings['policy_command_lower_rad'],settings['policy_command_upper_rad']);initial[6]=np.clip(initial[6],0,.1)
    epoch=time.monotonic();end=epoch+args.seconds
    class Guard:
        stop=SimpleNamespace(value=0);running=SimpleNamespace(value=0);policy_time=SimpleNamespace(value=epoch);startup_deadline=SimpleNamespace(value=end)
        def check(self):
            if time.monotonic()>=end:raise Finished()
            if self.stop.value:raise RuntimeError('Replay software guard stopped')
    class Cameras:
        def check(self):pass
    class Bank:
        def observation(self):
            from rtc_piper.observations import Observation
            t=epoch+np.floor((time.monotonic()-epoch)*30)/30-1/30
            return Observation(t,measured.copy(),images,np.full(4,t))
        def live(self):return measured.copy(),np.full(4,time.monotonic())
    class Arm:
        commands=0
        def status(self,**kwargs):pass
        def joints(self,x):
            if np.asarray(x).shape!=(6,) or not np.isfinite(x).all():raise RuntimeError('Nonfinite software command')
            self.commands+=1
        def grip(self,x,force):
            if not 0<=x<=.1:raise RuntimeError('Bad software gripper command')
    session=Session(args.mode);policy=Client(args.port,session);observer=Client(args.port,session);arm=Arm()
    args.root.mkdir(parents=True,exist_ok=True);(args.root/'logs').mkdir(exist_ok=True)
    report=dict(hardware_opened=False,robot_commands_sent=0,source='repeated saved images/state with synthetic replay timestamps',mode=args.mode)
    try:
        run(args.root,arm,Cameras(),Bank(),Guard(),policy,observer,initial,epoch-.2,settings)
    except Finished:report['status']='passed'
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        path=args.root/'logs/last_corrector_run.json'
        record=json.loads(path.read_text()) if path.exists() else {}
        policy_ms=record.get('policy_rtt_ms',[]);monitor_ms=record.get('monitor_rtt_ms',[])
        report.update(elapsed_s=time.monotonic()-epoch,software_joint_commands=arm.commands,
            stats=record.get('stats'),max_control_gap_ms=record.get('max_control_gap_ms'),
            policy_p50_ms=float(np.median(policy_ms)) if policy_ms else None,
            policy_p95_ms=float(np.percentile(policy_ms,95)) if policy_ms else None,
            monitor_p95_ms=float(np.percentile(monitor_ms,95)) if monitor_ms else None,
            worker_errors=record.get('worker_errors'))
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
