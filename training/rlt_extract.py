"""One of eight workers: frozen final-prefix features and aligned demo chunks."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from training.corrector_extract import decode,episodes
from rlt_piper.features import Extractor
from rlt_piper.contract import CHUNK,digest,SCHEMA,UPSTREAM

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--world-size',type=int,default=8)
    p.add_argument('--stride',type=int,default=6);a=p.parse_args()
    from openpi.policies import policy_config
    from rtc_piper.model_adapter import training_config,PROMPT
    policy=policy_config.create_trained_policy(training_config(a.output),a.checkpoint,default_prompt=PROMPT,sample_kwargs={'num_steps':10})
    extract=Extractor(policy);jobs=[[] for _ in range(a.world_size)];loads=[0]*a.world_size
    for row in sorted(episodes(a.data_root),key=lambda r:-r['frames']):
        i=int(np.argmin(loads));jobs[i].append(row);loads[i]+=row['frames']
    contract=dict(schema=SCHEMA,upstream=UPSTREAM,base_step=2500,base_frozen=True,stride=a.stride,chunk_len=CHUNK,
                  prefix='post-transformer two valid image views, 512x2048')
    a.output.mkdir(parents=True,exist_ok=True)
    for row in jobs[a.rank]:
        name='%s_%06d'%(row['partition'],row['episode']);target=a.output/name
        if (target/'complete.json').exists():
            if json.loads((target/'complete.json').read_text())['contract']!=contract:raise RuntimeError('Feature contract changed')
            continue
        partial=a.output/(name+'.partial');partial.mkdir(exist_ok=True)
        root=Path(row['root']);idx=row['episode'];n=row['frames']
        table=pq.read_table(root/('data/chunk-000/episode_%06d.parquet'%idx))
        state=np.asarray(table['observation.state'].to_pylist(),np.float32);action=np.asarray(table['action'].to_pylist(),np.float32)
        frames=[decode(root/('videos/chunk-000/observation.images.%s/episode_%06d.mp4'%(cam,idx))) for cam in ('cam0','cam1')]
        if state.shape!=action.shape or state.shape!=(n,7) or any(len(v)!=n for v in frames):raise RuntimeError('Episode alignment mismatch')
        anchors=np.arange(0,n-CHUNK+1,a.stride);prefix=None;refs=[]
        for j,i in enumerate(anchors):
            obs=dict(cam0=frames[0][i],cam1=frames[1][i],state=state[i],prompt=PROMPT)
            features=np.asarray(extract(obs),np.float16)
            if features.shape!=(512,2048) or not np.isfinite(features).all():raise RuntimeError('Invalid prefix features')
            if prefix is None:prefix=np.lib.format.open_memmap(partial/'prefix.npy',mode='w+',dtype=np.float16,shape=(len(anchors),512,2048))
            prefix[j]=features;refs.append(policy.infer(obs)['actions'][:CHUNK])
        prefix.flush();del prefix
        np.save(partial/'state.npy',state[anchors]);np.save(partial/'reference.npy',np.asarray(refs,np.float32))
        np.save(partial/'target.npy',np.stack([action[i:i+CHUNK] for i in anchors]));np.save(partial/'frame.npy',anchors)
        (partial/'complete.json').write_text(json.dumps(dict(**row,contract=contract,anchors=len(anchors)),indent=2)+'\n')
        partial.rename(target);print(json.dumps(dict(rank=a.rank,episode=name,anchors=len(anchors))),flush=True)

if __name__=='__main__':main()
