"""Physical command and artifact contracts for the native RLT adapter."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

SCHEMA='piper-openpi-rlt-v1'
UPSTREAM='c1e40ac360185778c98cf20da2820e22d2d415e7'
CHUNK=18

def upstream(root):
    path=Path(root)/'vendor/openpi_rlt'
    if not path.is_dir():raise RuntimeError('Pinned openpi-RLT sources missing: '+str(path))
    sys.path.insert(0,str(path))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(actions,state,stats):
    a=np.array(actions,dtype=np.float32,copy=True);s=np.asarray(state,dtype=np.float32)
    while s.ndim<a.ndim:s=s[...,None,:]
    a[...,:6]-=s[...,:6]
    lo=np.array(stats['q01'])[:7];hi=np.array(stats['q99'])[:7]
    return ((a-lo)/(hi-lo+1e-6)*2-1).astype(np.float32)

def absolute(actions,state,stats):
    s=np.asarray(state,dtype=np.float32);a=np.asarray(actions,dtype=np.float32)
    lo=np.array(stats['q01'])[:7];hi=np.array(stats['q99'])[:7]
    a=(a+1)*.5*(hi-lo+1e-6)+lo
    while s.ndim<a.ndim:s=s[...,None,:]
    a[...,:6]+=s[...,:6]
    if not np.isfinite(a).all():raise RuntimeError('RLT produced nonfinite actions')
    return a.astype(np.float32)

def assets(directory,actor=False):
    directory=Path(directory)
    required=['token.msgpack','token.json','norm_stats.json']
    missing=[str(directory/x) for x in required if not (directory/x).is_file()]
    if missing:raise RuntimeError('RLT assets are not trained/installed yet: '+', '.join(missing))
    m=json.loads((directory/'token.json').read_text())
    if (m.get('schema')!=SCHEMA or m.get('upstream')!=UPSTREAM or m.get('base_step')!=2500
            or m.get('base_frozen') is not True or m.get('trained') is not True
            or m.get('token_sha256')!=digest(directory/'token.msgpack')
            or m.get('norm_sha256')!=digest(directory/'norm_stats.json')):
        raise RuntimeError('RLT token/checkpoint/normalization identity mismatch')
    if actor:
        from rlt_piper.checkpoint import snapshot
        if snapshot(directory,m['token_sha256']) is None:
            raise RuntimeError('RLT actor is not trained/ready yet (untrained)')
    return m
