"""Saved-observation network comparison; no CAN/camera imports or commands."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .transport import Connection
from .wire import MODES,jpeg_pair,prepare_jpegs,pixel_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--port',type=int,default=21969)
    parser.add_argument('--runs',type=int,default=20);args=parser.parse_args()
    with np.load(args.sample,allow_pickle=False) as sample:
        images={k:np.array(sample[k]) for k in ('cam0','cam1')};state=np.array(sample['state'])
    expected=pixel_hash(prepare_jpegs(jpeg_pair(images)))
    report=dict(status='running',checkpoint_step=2250,denoising_steps=5,hardware_opened=False,robot_commands_sent=0,
        model_input_pixel_sha256=expected,route='collector -> desktop SSH relay -> existing SSH jump -> single H100',
        input='same saved raw640x480 RGB camera pair; encoded afresh every inference',modes={})
    reference=None
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    waiting=Connection(args.port,'json_keepalive')
    try:
        until=time.monotonic()+900;notice=0
        while time.monotonic()<until:
            try:waiting.health();break
            except (OSError,RuntimeError):pass
            if time.monotonic()>notice:print('Waiting for optimized single-H100 warmup...',flush=True);notice=time.monotonic()+20
            time.sleep(3)
        else:raise RuntimeError('Server readiness timed out')
    finally:waiting.close()
    try:
        for mode in MODES:
            client=Connection(args.port,mode);health=client.health()
            part=dict(status='running',warmups=[],runs=[],echo_runs=[]);report['modes'][mode]=part
            try:
                for i in range(args.runs+3):
                    actions,row,body,kind=client.infer(images,state,expected)
                    if reference is None:reference=actions.copy()
                    row.update(index=i,max_joint_difference_deg=float(np.degrees(np.max(abs(actions[:,:6]-reference[:,:6])))),
                        max_gripper_difference_mm=float(1000*np.max(abs(actions[:,6]-reference[:,6]))))
                    part['warmups' if i<3 else 'runs'].append(row);save()
                for i in range(20):part['echo_runs'].append(client.echo(body,kind));save()
                for key,rows,field in [('total',part['runs'],'total_ms'),('compute',part['runs'],'inference_ms'),('encode',part['runs'],'encode_ms'),('echo',part['echo_runs'],'total_ms')]:
                    values=[r[field] for r in rows]
                    part[key+'_ms']=dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),min=min(values),max=max(values))
                part.update(status='passed',connections_used=len({r['connection_id'] for r in part['runs']}),
                    median_request_bytes=float(np.median([r['request_bytes'] for r in part['runs']])))
                save();print(json.dumps(dict(mode=mode,**{k:v for k,v in part.items() if k.endswith('_ms') or k in ('connections_used','median_request_bytes')})),flush=True)
            finally:client.close()
        report['status']='passed';save();print('All network modes completed',flush=True)
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
