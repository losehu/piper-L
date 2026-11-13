"""Actual TX-time command traces, accepted plans and explicitly labelled outcomes."""
import json,threading,time,uuid
from pathlib import Path
import numpy as np
from rlt_piper.contract import SCHEMA,CHUNK

SOURCES={'base':0,'rlt':1,'corrector':2}

class Recorder:
    def __init__(self,root,stage,token_sha,settings):
        self.directory=Path(root)/'rlt_episodes'/('episode_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.stage,self.token_sha=stage,token_sha;self.plans=[];self.commands=[];self.label=None;self.label_at=None
        self.settings={k:settings[k] for k in ['fps','control_hz','execute_horizon','chunk_smoothing_steps','command_filter_tau_s']}
        self.lock=threading.Lock();self.error=None
    def outcome(self,value,guard):
        with self.lock:
            if self.label is None and not guard.stop.value:
                self.label=int(value);self.label_at=time.monotonic();guard.stop.value=1
    def plan(self,obs,result,received):
        item=result['rlt'];row=dict(t=float(obs.t),received=float(received),generation=int(result['generation']),
                                   source=SOURCES[item['source']],actor_version=int(item['actor_version']))
        row.update({k:np.array(item[k],copy=True) for k in ['z_rl','proprio','ref_chunk','proposed_chunk']})
        with self.lock:self.plans.append(row)
    def command(self,stamp,command):self.commands.append((float(stamp),np.array(command,copy=True)))
    def save(self,terminal=None,error=None):
        self.directory.mkdir(parents=True,exist_ok=False)
        meta=dict(schema=SCHEMA,stage=self.stage,token_sha256=self.token_sha,settings=self.settings,
                  labelled=self.label is not None,outcome=self.label,label_at=self.label_at,
                  terminal_available=terminal is not None,error=error,keyboard_error=self.error,
                  command_semantics='actually transmitted 120Hz rad/metre commands; timestamps recorded after TX')
        if self.plans and self.commands:
            values={k:np.array([p[k] for p in self.plans]) for k in self.plans[0]}
            values.update(command_t=np.array([t for t,_ in self.commands]),commands=np.stack([q for _,q in self.commands]))
            if terminal:
                values.update(terminal_t=np.array(terminal['t']),**{'terminal_'+k:np.asarray(terminal[k]) for k in ['z_rl','proprio','ref_chunk']})
            np.savez_compressed(self.directory/'trace.npz',**values)
        (self.directory/'episode.json').write_text(json.dumps(meta,indent=2)+'\n')
        return self.directory

def transitions(path,token_sha):
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
        result.append(row)
    return result
