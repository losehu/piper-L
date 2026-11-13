"""Actual RTC controller with delayed software policy and fake leader/follower."""
import argparse,json,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class Finished(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    from rtc_piper.observations import Observation
    from rlt_piper.control import run
    from rlt_piper.hil import Handover
    from rlt_piper.recording import Recorder
    cfg=select_model(config(a.root/'config.yaml'),'pi05',a.root)
    with np.load(a.root/'offline_sample.npz') as d:state=np.array(d['state']);images={k:np.array(d[k]) for k in ['cam0','cam1']}
    state[:6]=np.clip(state[:6],cfg['policy_command_lower_rad'],cfg['policy_command_upper_rad']);epoch=time.monotonic()
    h=Handover();rec=Recorder(a.output.parent/'unsaved_software_trace','online','synthetic',cfg)
    class Guard:
        stop=SimpleNamespace(value=0);running=SimpleNamespace(value=0)
        policy_time=SimpleNamespace(value=epoch);startup_deadline=SimpleNamespace(value=epoch+5)
        def check(self):
            if time.monotonic()-epoch>4:raise Finished()
            if self.stop.value:raise RuntimeError('Software guard stopped')
    class Bank:
        def observation(self):
            t=epoch+np.floor((time.monotonic()-epoch)*30)/30-1/30
            return Observation(t,state.copy(),images,np.full(4,t))
        def live(self):return state.copy(),np.full(4,time.monotonic())
    class Arm:
        def joints(self,q):
            if not np.isfinite(q).all():raise RuntimeError('Nonfinite software command')
        def grip(self,q,_):pass
        def status(self,**kw):pass
    class Leader:
        def vector(self):
            q=state.copy();q[0]+=.02*np.sin((time.monotonic()-epoch)*3)
            return q,np.full(4,time.monotonic())
    class Policy:
        def __init__(self):self.requests=[];self.closed=False
        def request(self,method,obs,generation=0,**kw):
            self.requests.append((generation,obs.t));time.sleep(.14)
            actions=np.tile(state,(50,1));actions[:,0]+=.015
            return dict(generation=generation,actions=actions,rlt=dict(z_rl=np.zeros(512),proprio=state.copy(),
                ref_chunk=actions[:18],proposed_chunk=actions[:18],source='base',actor_version=0))
        def close(self):self.closed=True
    def keys():
        for deadline in [.3,1.7,2.4,3.1]:
            time.sleep(max(0,epoch+deadline-time.monotonic()));h.toggle()
    thread=threading.Thread(target=keys);thread.start();policy=Policy()
    try:run(a.root,Arm(),SimpleNamespace(check=lambda:None),Bank(),Guard(),policy,state.copy(),epoch-.2,cfg,rec,h,Leader())
    except Finished:pass
    finally:thread.join()
    if len(rec.events)!=4 or not any(c[2]==2 for c in rec.commands):raise RuntimeError('Handover was not exercised')
    if not all(p['t']>next(e['t'] for e in rec.events if e['generation']==p['generation']) for p in rec.plans if p['generation']):raise RuntimeError('Stale chunk accepted')
    # At least one in-flight request crossed each handover; none may be accepted afterwards.
    for p in rec.plans:
        if any(e['generation']>p['generation'] and e['t']<p['received'] for e in rec.events):raise RuntimeError('Old generation accepted')
    gaps=np.diff([c[0] for c in rec.commands])
    report=dict(status='passed',hardware_opened=False,robot_commands_sent=0,saved_formal_data=False,
                handovers=len(rec.events),accepted_plans=len(rec.plans),software_commands=len(rec.commands),max_control_gap_ms=float(max(gaps)*1000))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
