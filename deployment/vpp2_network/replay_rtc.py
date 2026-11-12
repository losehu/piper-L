"""Actual remote RPC + synthetic120Hz action replay, with no robot devices.

Saved images/state are repeated. Request times minus a declared reserve model
camera age for timing checks; these are replay anchors, not live camera stamps.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from rtc_piper.cli import config
from rtc_piper.profiles import select_model
from rtc_piper.observations import Observation,NotReady
from rtc_piper.hardware import follower_lock
from rtc_piper.policy import CommandSmoother
from rtc_piper.vpp2_rtc import CaptureTimeline,Producer
from .transport import PolicyClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--requests',type=int,default=60)
    parser.add_argument('--port',type=int,default=21969);parser.add_argument('--camera-age',type=float,default=1/30)
    parser.add_argument('--inflight',type=int,choices=[1,2],default=2)
    args=parser.parse_args();base=config(args.config);base.setdefault('vpp2',{})['inference']='h100_async_rtc'
    base['vpp2']['max_inflight']=args.inflight
    settings=select_model(base,'vpp2',args.config.parent)
    with follower_lock(settings):run(args,settings)


def run(args,settings):
    with np.load(args.sample,allow_pickle=False) as data:
        state=np.array(data['state']);images={k:np.array(data[k]) for k in ('cam0','cam1')}
    initial=state.copy();initial[:6]=np.clip(initial[:6],settings['policy_command_lower_rad'],settings['policy_command_upper_rad']);initial[6]=np.clip(initial[6],0,.1)
    client=PolicyClient(args.port,settings);timeline=CaptureTimeline(initial,settings);smoother=CommandSmoother(initial,settings)
    guard=SimpleNamespace(stop=SimpleNamespace(value=0),policy_time=SimpleNamespace(value=0))
    epoch=time.monotonic()
    def observation():
        if producer.stats['requests']>=args.requests:raise NotReady('Replay request count reached')
        anchor=epoch+np.floor((time.monotonic()-epoch)*30)/30-args.camera_age
        return Observation(anchor,state.copy(),images,np.full(4,anchor))
    producer=Producer(client,observation,lambda:state,timeline,guard,settings)
    report=dict(status='running',hardware_opened=False,robot_commands_sent=0,
        observation_source='repeated saved images/state; synthetic replay anchors, not live camera capture',
        camera_age_reserve_s=args.camera_age,request_target=args.requests,control_target_hz=120)
    producer.thread.start();begun=time.monotonic();started=None;previous_time=begun;dts=[];error=None;loops=0
    try:
        while not (producer.stats['requests']>=args.requests and producer.stats['inflight']==0):
            now=time.monotonic()
            if producer.error:raise RuntimeError(str(producer.error)) from producer.error
            if now-begun>args.requests*3+10:raise RuntimeError('Replay timeout')
            target=timeline.sample(now)
            if target is None:
                if started is not None:raise RuntimeError('Replay action window exhausted')
                previous_time=now
            else:
                if started is None:started=now;previous_time=now
                dt=max(now-previous_time,1e-6);dts.append(dt)
                command=smoother.step(target,dt)
                if not np.isfinite(command).all():raise RuntimeError('Invalid replay command')
                loops+=1;previous_time=now
            time.sleep(max(0,1/120-(time.monotonic()-now)))
        report['status']='passed'
    except BaseException as exc:error=repr(exc);report['status']='failed';raise
    finally:
        guard.stop.value=1;client.close();producer.thread.join(.5)
        events=list(producer.events);overlaps=[e['overlap_remaining_s'] for e in events if e['overlap_remaining_s'] is not None]
        values=1000*np.asarray(producer.latencies)
        report.update(error=error,requests=producer.stats,elapsed_s=time.monotonic()-begun,control_loops=loops,
            last_transport_stats=client.last_stats,
            max_control_gap_ms=1000*max(dts,default=0),p95_control_gap_ms=1000*float(np.percentile(dts,95)) if dts else None,
            request_p50_ms=float(np.median(values)) if len(values) else None,
            request_p95_ms=float(np.percentile(values,95)) if len(values) else None,
            min_overlap_remaining_ms=1000*min(overlaps) if overlaps else None,events=events)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='events'},indent=2),flush=True)


if __name__=='__main__':main()
