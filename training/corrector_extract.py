"""One of eight GPU workers: frame-aligned frozen-encoder feature extraction."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import av
import numpy as np
import pyarrow.parquet as pq
from corrector.contract import identity,action_normalize
from corrector.vision import load_encoder


def decode(path):
    with av.open(str(path)) as video:
        video.streams.video[0].thread_type='AUTO'
        return np.stack([frame.to_ndarray(format='rgb24') for frame in video.decode(video=0)])


def episodes(data_root):
    result=[]
    for partition in ('train','val'):
        root=Path(data_root)/'local'/('piper_grasp_'+partition)
        rows=[json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines() if s]
        for row in rows:result.append(dict(root=str(root),partition=partition,episode=int(row['episode_index']),frames=int(row['length'])))
    # Greedy balancing by frames; each complete episode belongs to one worker.
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--encoder',type=Path,required=True)
    p.add_argument('--norm',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=8);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--k-step',type=int,default=5);args=p.parse_args()
    import jax
    import jax.numpy as jnp
    if len(jax.devices('gpu'))!=1:raise RuntimeError('Expose exactly one GPU per extractor')
    norms=json.loads(args.norm.read_text());norms=norms.get('norm_stats',norms)
    contract=identity(norms,args.k_step)
    args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    jobs=[[] for _ in range(args.world_size)];loads=[0]*args.world_size
    for row in sorted(episodes(args.data_root),key=lambda x:-x['frames']):
        i=int(np.argmin(loads));jobs[i].append(row);loads[i]+=row['frames']
    encoder=load_encoder(args.encoder);done=0
    for row in jobs[args.rank]:
        name='%s_%06d'%(row['partition'],row['episode']);target=args.output/name
        if (target/'complete.json').exists():
            if json.loads((target/'complete.json').read_text())['contract']!=contract:raise RuntimeError('Existing feature contract changed')
            continue
        partial=args.output/(name+'.partial');partial.mkdir(exist_ok=True)
        root=Path(row['root']);episode=row['episode'];n=row['frames']
        table=pq.read_table(root/('data/chunk-000/episode_%06d.parquet'%episode))
        states=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        actions=np.asarray(table['action'].to_pylist(),dtype=np.float32)
        assert states.shape==actions.shape==(n,7)
        normalized=action_normalize(actions,states,norms['actions'])
        np.save(partial/'actions.npy',normalized,allow_pickle=False)
        cameras=[decode(root/('videos/chunk-000/observation.images.%s/episode_%06d.mp4'%(cam,episode))) for cam in ('cam0','cam1')]
        if any(a.shape!=(n,224,224,3) for a in cameras):raise RuntimeError('Video frame count/shape mismatch')
        features=None
        for begin in range(0,n,args.batch):
            end=min(begin+args.batch,n);size=end-begin
            # One encoder call, then restore base/wrist order within each frame.
            batch=np.concatenate([a[begin:end] for a in cameras],axis=0)
            if size<args.batch:batch=np.concatenate([batch,np.repeat(batch[-1:],2*(args.batch-size),axis=0)])
            z=np.asarray(encoder(jnp.asarray(batch,dtype=jnp.float32)/255*2-1,train=False)[0])
            z=np.concatenate([z[:size],z[size:2*size]],axis=1).astype(np.float16)
            if not np.isfinite(z).all():raise RuntimeError('Nonfinite visual features')
            if features is None:features=np.lib.format.open_memmap(partial/'z.npy',mode='w+',dtype=np.float16,shape=(n,*z.shape[1:]))
            features[begin:end]=z
        features.flush();shape=list(features.shape);del features,cameras
        (partial/'complete.json').write_text(json.dumps(dict(**row,contract=contract,shape=shape,rank=args.rank),indent=2)+'\n')
        partial.rename(target);done+=n
        print(json.dumps(dict(rank=args.rank,episode=name,frames=n,completed_frames=done,elapsed_s=time.monotonic()-started)),flush=True)
        gc.collect()
    (args.output/('rank%d.done.json'%args.rank)).write_text(json.dumps(dict(rank=args.rank,frames=done,assigned_frames=loads[args.rank],contract=contract)))


if __name__=='__main__':main()
