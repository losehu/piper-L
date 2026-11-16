"""Actual RTC controller with delayed software policy and fake leader/follower."""
import argparse,json,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

class Finished(Exception):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--lagged-servo',action='store_true');a=p.parse_args()
    from rtc_piper.cli import config
    from rtc_piper.profiles import select_model
    from rtc_piper.observations import Observation
    from rlt_piper.control import run
    from rlt_piper.hil import Handover
    from rlt_piper.recording import Recorder
    cfg=select_model(config(a.root/'config.yaml'),'pi05',a.root)
    with np.load(a.root/'offline_sample.npz') as d:state=np.array(d['state']);images={k:np.array(d[k]) for k in ['cam0','cam1']}
    state[:6]=np.clip(state[:6],cfg['policy_command_lower_rad'],cfg['policy_command_upper_rad']);epoch=time.monotonic()
    servo=SimpleNamespace(target=state[:6].copy(),velocity=np.zeros(6),updated=epoch,lock=threading.Lock())
    def measured():
        if a.lagged_servo:
            with servo.lock:
                now=time.monotonic();remaining=now-servo.updated
                while remaining>1e-9:
                    dt=min(.001,remaining);servo.velocity+=(144*(servo.target-state[:6])-24*servo.velocity)*dt;state[:6]+=servo.velocity*dt;remaining-=dt
                servo.updated=now
        return state.copy()
    h=Handover();rec=Recorder(a.output.parent/'unsaved_software_trace','online','synthetic',cfg)
    class Guard:
        stop=SimpleNamespace(value=0);running=SimpleNamespace(value=0)
        policy_time=SimpleNamespace(value=epoch);startup_deadline=SimpleNamespace(value=epoch+5)
        def check(self):
            if time.monotonic()-epoch>(8 if a.lagged_servo else 4):raise Finished()
            if self.stop.value:raise RuntimeError('Software guard stopped')
    class Bank:
        def observation(self):
            t=epoch+np.floor((time.monotonic()-epoch)*30)/30-1/30
            return Observation(t,measured(),images,np.full(4,t))
        def live(self):return measured(),np.full(4,time.monotonic())
    class Arm:
        def joints(self,q):
            if not np.isfinite(q).all():raise RuntimeError('Nonfinite software command')
            if a.lagged_servo:
                measured()
                with servo.lock:servo.target=np.array(q,copy=True)
        def grip(self,q,_):pass
        def status(self,**kw):pass
    class Leader:
        def vector(self):
            q=measured();q[0]+=.02*np.sin((time.monotonic()-epoch)*3)
            return q,np.full(4,time.monotonic())
    class Policy:
        def __init__(self):self.requests=[];self.closed=False
        def request(self,method,obs,generation=0,**kw):
            self.requests.append((generation,obs.t));time.sleep(.14)
            actions=np.tile(measured(),(50,1));actions[:,0]+=.08 if a.lagged_servo else .015
            return dict(generation=generation,actions=actions,rlt=dict(z_rl=np.zeros(512),proprio=state.copy(),
                ref_chunk=actions[:18],proposed_chunk=actions[:18],source='base',actor_version=0))
        def close(self):self.closed=True
    class FakeAligner:
        error=None;phase='idle';progress=0.
        def start(self,target):self.target=np.array(target);self.began=time.monotonic();self.phase='moving'
        @property
        def result(self):
            if not hasattr(self,'began') or time.monotonic()-self.began<.12:return None
            self.phase='ready';return self.target.copy(),np.full(4,time.monotonic())
    h.aligner=FakeAligner()
    def keys():
        for deadline in ([.6,2.7,4.,6.4] if a.lagged_servo else [.3,1.7,2.4,3.1]):
            time.sleep(max(0,epoch+deadline-time.monotonic()));h.toggle()
    thread=threading.Thread(target=keys);thread.start();policy=Policy()
    try:run(a.root,Arm(),SimpleNamespace(check=lambda:None),Bank(),Guard(),policy,state.copy(),epoch-.2,cfg,rec,h,Leader())
    except Finished:pass
    finally:thread.join()
    if len(rec.events)!=7 or not any(c[2]==2 for c in rec.commands):raise RuntimeError('Handover was not exercised')
    if not all(p['t']>next(e['t'] for e in rec.events if e['generation']==p['generation']) for p in rec.plans if p['generation']):raise RuntimeError('Stale chunk accepted')
    # At least one in-flight request crossed each handover; none may be accepted afterwards.
    for p in rec.plans:
        if any(e['generation']>p['generation'] and e['t']<p['received'] for e in rec.events):raise RuntimeError('Old generation accepted')
    for generation in [e['generation'] for e in rec.events if e['mode']=='aligning']:
        holds=[c[1] for c in rec.commands if c[2]==5 and c[3]==generation]
        if len(holds)<50 or np.max(np.ptp(np.array(holds),axis=0))>1e-5:raise RuntimeError('Follower hold command changed during leader alignment')
    gaps=np.diff([c[0] for c in rec.commands])
    report=dict(status='passed',hardware_opened=False,robot_commands_sent=0,saved_formal_data=False,
                lagged_servo=a.lagged_servo,handovers=len(rec.events),accepted_plans=len(rec.plans),software_commands=len(rec.commands),max_control_gap_ms=float(max(gaps)*1000))
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
