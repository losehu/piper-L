"""Immutable learner generations, published by one atomic pointer replacement."""
import json,pickle,time,uuid
from pathlib import Path
from rlt_piper.contract import SCHEMA,CHUNK,digest

def current(directory):
    directory=Path(directory);pointer=directory/'CURRENT'
    if not pointer.exists():return directory
    name=pointer.read_text().strip()
    if not name.startswith('step_') or Path(name).name!=name:raise RuntimeError('Invalid learner pointer')
    path=directory/'generations'/name
    if not path.is_dir():raise RuntimeError('Missing published learner generation')
    return path

def publish(directory,state_bytes,payload,info):
    directory=Path(directory);parent=directory/'generations';parent.mkdir(exist_ok=True)
    name='step_%09d_%s'%(info['global_step'],uuid.uuid4().hex[:8]);path=parent/name;path.mkdir()
    (path/'learner.msgpack').write_bytes(state_bytes)
    (path/'actor.pkl').write_bytes(pickle.dumps(payload,protocol=5))
    meta=dict(info,schema=SCHEMA,chunk_len=CHUNK,actor_sha256=digest(path/'actor.pkl'),time=time.time())
    (path/'actor.json').write_text(json.dumps(meta,indent=2)+'\n')
    temp=directory/'CURRENT.partial';temp.write_text(name+'\n');temp.replace(directory/'CURRENT')
    return path

def snapshot(directory,token_sha,require_ready=True):
    path=current(directory)
    if not (path/'actor.json').exists():return None
    meta=json.loads((path/'actor.json').read_text())
    if (meta.get('schema')!=SCHEMA or meta.get('token_sha256')!=token_sha or meta.get('chunk_len')!=CHUNK
            or meta.get('actor_sha256')!=digest(path/'actor.pkl')):raise RuntimeError('Actor checkpoint identity mismatch')
    if require_ready and (meta.get('actor_updates',0)<1 or not meta.get('ready',True)):return None
    with (path/'actor.pkl').open('rb') as f:payload=pickle.load(f)
    if payload.get('token_sha256')!=token_sha or int(payload['version'])!=meta['actor_updates']:raise RuntimeError('Actor payload version mismatch')
    return payload,meta
