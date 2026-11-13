"""Atomic live replay snapshots with TX provenance and separate task outcomes."""
import json,threading,time,uuid
from pathlib import Path
import numpy as np
from rlt_piper.contract import SCHEMA,CHUNK

REPLAY='piper-rlt-hil-v2'
SOURCES={'base':0,'rlt':1,'human':2,'corrector':4,'hold':5}

class Recorder:
    def __init__(self,root,stage,token_sha,settings):
        self.directory=Path(root)/'rlt_episodes'/('episode_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.stage,self.token_sha=stage,token_sha;self.plans=[];self.commands=[];self.label=None;self.label_at=None
        self.settings={k:settings[k] for k in ['fps','control_hz','execute_horizon','chunk_smoothing_steps','command_filter_tau_s']}
        self.lock=threading.Lock();self.write_lock=threading.Lock();self.error=None;self.revision=0
        self.closed=threading.Event();self.events=[]
    def outcome(self,value,guard):
        with self.lock:
            if self.label is None and not guard.stop.value:
                self.label=int(value);self.label_at=time.monotonic();guard.stop.value=1
    def plan(self,obs,result,received,source=None):
        item=result['rlt'];row=dict(t=float(obs.t),received=float(received),generation=int(result['generation']),
                                   source=SOURCES[source or item['source']],actor_version=int(item['actor_version']))
        row['state_times']=np.array(getattr(obs,'state_times',np.full(4,np.nan)),copy=True)
        row.update({k:np.array(item[k],copy=True) for k in ['z_rl','proprio','ref_chunk','proposed_chunk']})
        with self.lock:self.plans.append(row)
    def command(self,stamp,command,source=None,generation=0,leader=None,leader_times=None):
        row=(float(stamp),np.array(command,copy=True),-1 if source is None else SOURCES[source],int(generation),
             np.full(7,np.nan) if leader is None else np.array(leader,copy=True),
             np.full(4,np.nan) if leader_times is None else np.array(leader_times,copy=True))
        with self.lock:self.commands.append(row)
    def event(self,mode,generation,stamp):
        with self.lock:self.events.append(dict(mode=mode,generation=generation,t=stamp))
    def start(self):
        self.directory.mkdir(parents=True,exist_ok=False)
        self.thread=threading.Thread(target=self._journal,daemon=True);self.thread.start()
    def _journal(self):
        try:
            while not self.closed.wait(2):self.snapshot(live=True)
        except BaseException as e:self.error='Live journal: '+repr(e)
    def snapshot(self,terminal=None,error=None,live=False):
        with self.write_lock:
            self.directory.mkdir(parents=True,exist_ok=True)
            with self.lock:
                plans=list(self.plans);commands=list(self.commands);events=list(self.events)
                label,label_at=self.label,self.label_at
            self.revision+=1;name='trace_%08d.npz'%self.revision
            meta=dict(schema=SCHEMA,replay_schema=REPLAY,stage=self.stage,token_sha256=self.token_sha,settings=self.settings,
                      labelled=label is not None,outcome=label,label_at=label_at,live=live,events=events,
                      terminal_available=terminal is not None,error=error,keyboard_error=self.error,trace=name,
                      revision=self.revision,command_semantics='actually transmitted quantized 120Hz rad/metre commands; post-TX timestamps')
            if plans and commands:
                values={k:np.array([p[k] for p in plans]) for k in plans[0]}
                values.update(command_t=np.array([c[0] for c in commands]),commands=np.stack([c[1] for c in commands]),
                              command_source=np.array([c[2] for c in commands]),command_generation=np.array([c[3] for c in commands]),
                              leader=np.stack([c[4] for c in commands]),leader_times=np.stack([c[5] for c in commands]))
                if terminal:values.update(terminal_t=np.array(terminal['t']),**{'terminal_'+k:np.asarray(terminal[k]) for k in ['z_rl','proprio','ref_chunk']})
                np.savez_compressed(self.directory/name,**values)
            else:meta['trace']=None
            temp=self.directory/'episode.json.partial';temp.write_text(json.dumps(meta,indent=2)+'\n');temp.replace(self.directory/'episode.json')
            # Reader retries manifest if a previous revision has been reclaimed.
            for old in self.directory.glob('trace_*.npz'):
                if int(old.stem.split('_')[1])<self.revision-2:old.unlink(missing_ok=True)
            return self.directory
    def save(self,terminal=None,error=None):
        self.closed.set()
        if hasattr(self,'thread'):self.thread.join(10)
        if hasattr(self,'thread') and self.thread.is_alive():raise RuntimeError('Replay writer did not stop')
        path=self.snapshot(terminal,error)
        # Compatibility link for inspection tools; learner uses manifest only.
        meta=json.loads((path/'episode.json').read_text())
        if meta['trace']:
            link=path/'trace.npz';link.unlink(missing_ok=True);link.symlink_to(meta['trace'])
        return path

def transitions(path,token_sha):
    path=Path(path)
    for _ in range(3):
        m=json.loads((path/'episode.json').read_text())
        if m.get('replay_schema')!=REPLAY:return _legacy_transitions(path,token_sha)
        if m.get('token_sha256')!=token_sha or not m.get('trace') or m.get('keyboard_error'):return []
        try:
            with np.load(path/m['trace'],allow_pickle=False) as data:d={k:data[k] for k in data.files}
            break
        except FileNotFoundError:continue
    else:return []
    times=d['t'];cmd_t=d['command_t'];result=[]
    terminal=bool(m.get('labelled') and m.get('terminal_available') and not m.get('error'))
    end=min(float(m['label_at']),float(d['terminal_t'])) if terminal else cmd_t[-1]
    for i,t in enumerate(times):
        valid=min(CHUNK,int(np.ceil((end-t)*30-1e-7)))
        if valid<1 or (not terminal and valid<CHUNK):continue
        grid=t+np.arange(valid)/30;indices=np.searchsorted(cmd_t,grid,side='right')-1
        if np.any(indices<0) or np.any(grid-cmd_t[indices]>.1):continue
        source=d['command_source'][indices];generation=d['command_generation'][indices]
        # Compatibility for software replays created without explicit TX ownership.
        if np.all(source==-1):
            owners=np.searchsorted(d['received'],grid,side='right')-1
            if np.any(owners<0):continue
            source=d['source'][owners];generation=d['generation'][owners]
        if np.any(~np.isin(source,[0,1,2])) or np.any(generation!=d['generation'][i]):continue
        done=terminal and t+CHUNK/30>=end-1e-6
        j=np.searchsorted(times,t+CHUNK/30)
        if not done:
            if j>=len(times) or times[j]-(t+CHUNK/30)>.15 or d['generation'][j]!=d['generation'][i]:continue
            following={k:d[k][j] for k in ['z_rl','proprio','ref_chunk']}
        else:following={k:d['terminal_'+k] for k in ['z_rl','proprio','ref_chunk']}
        # Unlabelled policy experience can bootstrap while live. After an abort,
        # retain only corrections; never invent terminal rewards for an abort.
        if not terminal and not m.get('live') and not np.all(source==2):continue
        action=d['commands'][indices];padded=np.repeat(action[-1:],CHUNK,axis=0);padded[:valid]=action
        rewards=np.zeros(CHUNK,np.float32)
        if done:rewards[valid-1]=m['outcome']
        sources=np.repeat(source[-1],CHUNK).astype(np.int32);sources[:valid]=source
        result.append(dict(z_rl=d['z_rl'][i],proprio=d['proprio'][i],ref_chunk=d['ref_chunk'][i],action_chunk=padded,
            rewards=rewards,done=np.float32(done),source_chunk=sources,valid_steps=valid,
            transition_id='%s:%.9f'%(path.name,t),labelled=terminal,**{'next_'+k:v for k,v in following.items()}))
    return result
def _legacy_transitions(path,token_sha):
    """Reconstruct executed30Hz windows; exclude recovery/generation boundaries.

    Terminal windows retain their valid length; repeated tail entries are a
    model-input padding convention, never represented as transmitted commands.
    """
    path=Path(path);m=json.loads((path/'episode.json').read_text())
    if (not (path/'trace.npz').is_file() or m.get('schema')!=SCHEMA or m.get('token_sha256')!=token_sha or not m.get('labelled')
            or not m.get('terminal_available') or m.get('keyboard_error')):return []
    d=np.load(path/'trace.npz',allow_pickle=False);times=d['t'];cmd_t=d['command_t']
    # Terminal features are sampled after damping. Executed action time ends
    # at the label, not at that later observation; never invent extra actions.
    end=min(float(m['label_at']),float(d['terminal_t']))
    result=[]
    for i,t in enumerate(times):
        valid=min(CHUNK,int(np.ceil((end-t)*30-1e-7)))
        if valid<1:continue
        grid=t+np.arange(valid)/30;indices=np.searchsorted(cmd_t,grid,side='right')-1
        if np.any(indices<0) or np.any(grid-cmd_t[indices]>.1):continue
        owners=np.searchsorted(d['received'],grid,side='right')-1
        if np.any(owners<0) or np.any(d['source'][owners]==2) or np.any(d['generation'][owners]!=d['generation'][i]):continue
        done=t+CHUNK/30>=end-1e-6
        j=np.searchsorted(times,t+CHUNK/30)
        if j<len(times) and not done:
            if times[j]-(t+CHUNK/30)>.15 or d['generation'][j]!=d['generation'][i]:continue
            following={k:d[k][j] for k in ['z_rl','proprio','ref_chunk']}
        elif done:following={k:d['terminal_'+k] for k in ['z_rl','proprio','ref_chunk']}
        else:continue
        action=d['commands'][indices];padded=np.repeat(action[-1:],CHUNK,axis=0);padded[:valid]=action
        rewards=np.zeros(CHUNK,np.float32)
        if done:rewards[valid-1]=m['outcome']
        source=np.zeros(CHUNK,np.int32);source[:valid]=d['source'][owners];source[valid:]=source[valid-1]
        row=dict(z_rl=d['z_rl'][i],proprio=d['proprio'][i],ref_chunk=d['ref_chunk'][i],action_chunk=padded,
                 rewards=rewards,done=np.float32(done),source_chunk=source,valid_steps=valid,
                 **{'next_'+k:v for k,v in following.items()})
        row.update(transition_id='%s:%.9f'%(path.name,t),labelled=True)
        result.append(row)
    return result
