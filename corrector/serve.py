"""Dedicated, model-only OpenPI/JAX Corrector service; no hardware imports."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from corrector.contract import VERSION,identity,action_normalize
from corrector.monitor import LatentMonitor


class Service:
    def __init__(self,policy,metadata,norms,eta=1.):
        self.policy,self.metadata,self.norms,self.eta=policy,metadata,norms,eta
        self.session=None;self.mode=None;self.failed=False;self.timings=[]

    def adopt(self,session,mode):
        if not isinstance(session,str) or not 8<=len(session)<=64 or mode not in ('monitor','truncate','full'):
            raise ValueError('Invalid Corrector session/mode')
        if session!=self.session:
            self.session=session;self.mode=mode;self.generation=0;self.event_id=0;self.pending_guide=False
            self.monitor=LatentMonitor(self.policy.predict,self.metadata['contract']['k_step'],self.metadata['monitor_config'])
        elif mode!=self.mode:raise ValueError('Cannot change Corrector mode within an active session')

    def envelope(self,request):
        event=self.monitor.last_event
        return dict(protocol=VERSION,request_id=request['request_id'],session=self.session,
            mode=self.mode,generation=self.generation,event_id=self.event_id,
            event_at=None if event is None else event['stamp'],
            monitor_stats=dict(self.monitor.stats),robot_connected=False)

    def observe(self,z,request):
        command=np.asarray(request['executed_command'],dtype=np.float32)
        state=np.asarray(request['state'],dtype=np.float32)
        if command.shape!=(7,) or state.shape!=(7,):raise ValueError('Expected7D actual command and state')
        action=action_normalize(command,state,self.norms['actions'])
        result=self.monitor.observe(z,action,float(request['observed_at']))
        if result.get('event'):
            self.event_id+=1
            if self.mode!='monitor':self.generation+=1;self.pending_guide=self.mode=='full'
        return {k:v for k,v in result.items() if k!='target'}

    def infer(self,request):
        if self.failed:raise RuntimeError('Corrector policy error latched; restart service')
        try:return self._infer(request)
        except BaseException:self.failed=True;raise

    def _infer(self,request):
        start=time.monotonic();self.adopt(request['session'],request['mode'])
        method=request.get('method','infer')
        if method=='status':return dict(self.envelope(request),ok=True)
        if method not in ('infer','observe'):raise ValueError('Unknown Corrector method')
        observed=float(request['observed_at'])
        if not np.isfinite(observed):raise ValueError('Invalid capture time')
        obs={k:request[k] for k in ('cam0','cam1','state','prompt')}
        if method=='observe':
            z=self.policy.encode(obs);monitor=self.observe(z,request)
            return dict(self.envelope(request),ok=True,monitor=monitor,elapsed_ms=1000*(time.monotonic()-start))
        generation=int(request['generation'])
        if generation!=self.generation:
            return dict(self.envelope(request),ok=True,needs_replan=True,actions=None)
        guided=bool(self.pending_guide and self.mode=='full')
        target=self.monitor.last_event['target'] if guided else None
        actions,z,_=self.policy.infer(obs,target=target,enabled=guided,eta=self.eta)
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise RuntimeError('Invalid pi05 action output')
        monitor=self.observe(z,request)
        stale=generation!=self.generation
        if guided and not stale:self.pending_guide=False
        elapsed=1000*(time.monotonic()-start);self.timings.append(elapsed)
        return dict(self.envelope(request),ok=True,needs_replan=stale,actions=None if stale else actions,
            guided=guided,observed_at=observed,monitor=monitor,elapsed_ms=elapsed)


def load(root,assets):
    import jax
    from openpi.policies import policy_config
    from rtc_piper.model_adapter import training_config,PROMPT
    from corrector.jax_mlp import load as load_mlp
    from corrector.pi05_jax import GuidedPolicy
    if jax.default_backend()!='gpu':raise RuntimeError('Corrector deployment requires GPU; CPU fallback is disabled')
    meta=json.loads((assets/'metadata.json').read_text());norms=json.loads((assets/'norm_stats.json').read_text())
    norms=norms.get('norm_stats',norms)
    if meta['contract']!=identity(norms,meta['contract']['k_step']):raise ValueError('Corrector identity/normalization mismatch')
    if meta['metrics']['val_cosine_loss']>=1:raise RuntimeError('Corrector did not beat zero-residual direction baseline')
    checkpoint=root/'checkpoints/pi05_piper_2500_bf16'
    base_norms=json.loads((checkpoint/'assets/local/piper_grasp_train/norm_stats.json').read_text())
    base_norms=base_norms.get('norm_stats',base_norms)
    if identity(base_norms,meta['contract']['k_step'])!=meta['contract']:raise ValueError('Base pi05 and Corrector normalization differ')
    print('Loading frozen pi05 and Corrector...',flush=True)
    original=policy_config.create_trained_policy(training_config(root),checkpoint,default_prompt=PROMPT,sample_kwargs={'num_steps':10})
    guided=GuidedPolicy(original,load_mlp(assets/'mlp.npz'),guidance_every=5)
    with np.load(root/'offline_sample.npz',allow_pickle=False) as sample:obs={k:np.array(sample[k]) for k in ('cam0','cam1','state','prompt')}
    from openpi_client.image_tools import resize_with_pad
    obs['prompt']=str(obs['prompt'].item())
    for key in ('cam0','cam1'):obs[key]=resize_with_pad(obs[key],224,224)
    noise=np.random.default_rng(42).normal(size=(50,32)).astype(np.float32)
    print('Checking original/guided-sampler unguided parity...',flush=True)
    reference=original.infer(dict(obs),noise=noise)['actions']
    actual,z,_=guided.infer(obs,noise=noise)
    difference=np.max(abs(reference-actual),axis=0)
    # Sampling algebra must preserve the unguided policy within BF16/JIT tolerance.
    print('Unguided max physical differences: '+str(difference),flush=True)
    if np.max(difference[:6])>1e-5 or difference[6]>1e-6:raise RuntimeError('Unguided OpenPI parity exceeded tolerance: '+str(difference))
    target=guided.predict(z,np.zeros(7,dtype=np.float32))
    warm=[]
    guided_joint_delta=0.;guided_grip_delta=0.
    for enabled in (False,True,False,True,False):
        before=time.monotonic();a,_,_=guided.infer(obs,target=target,enabled=enabled,noise=noise)
        if not np.isfinite(a).all():raise RuntimeError('Nonfinite guided warmup')
        if enabled:
            guided_joint_delta=max(guided_joint_delta,float(np.max(abs(a[:,:6]-actual[:,:6]))))
            guided_grip_delta=max(guided_grip_delta,float(np.max(abs(a[:,6]-actual[:,6]))))
        warm.append(dict(guided=enabled,ms=1000*(time.monotonic()-before)))
    zero_eta,_,_=guided.infer(obs,target=target,enabled=True,eta=0.,noise=noise)
    if not np.array_equal(zero_eta,actual):raise RuntimeError('Zero guidance must exactly preserve baseline output')
    if max(guided_joint_delta,guided_grip_delta)==0:raise RuntimeError('Guidance warmup had no measurable action effect')
    # The observer has its own compiled vision path, and ordinary online calls
    # generate noise rather than receiving the fixed parity-test noise.
    for _ in range(2):
        monitor_z=guided.encode(obs);guided.predict(monitor_z,np.zeros(7,dtype=np.float32))
        guided.infer(obs,enabled=False)
    metadata=dict(protocol=VERSION,model='pi05_piper_grasp_2500_corrector',training_updates=2500,source_checkpoint_index=2499,
        action_horizon=50,physical_action_dim=7,fps=30,warmed_up=True,robot_connected=False,backend=jax.default_backend(),
        corrector_parameters=meta['parameters'],corrector_epoch=meta['epoch'],contract=meta['contract'],monitor_config=meta['monitor_config'],
        unguided_max_joint_difference_rad=float(max(difference[:6])),unguided_max_gripper_difference_m=float(difference[6]),
        warmups=warm,guide_steps_per_recovery=2)
    metadata.update(zero_eta_exact=True,guided_max_joint_delta_rad=guided_joint_delta,guided_max_gripper_delta_m=guided_grip_delta)
    return guided,metadata,norms


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--port',type=int,default=8002);p.add_argument('--eta',type=float,default=1.);args=p.parse_args()
    root=args.root.resolve();assets=args.assets.resolve();logs=root/'logs';logs.mkdir(exist_ok=True)
    argv=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
    state=dict(pid=os.getpid(),argv=argv,status='loading',owner_root=str(root),hardware_opened=False,assets=str(assets),port=args.port)
    path=logs/'corrector_policy_state.json';path.write_text(json.dumps(state,indent=2)+'\n')
    policy,metadata,norms=load(root,assets);state.update(status='ready',metadata=metadata)
    path.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2),flush=True)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(Service(policy,metadata,norms,args.eta),host='127.0.0.1',port=args.port,metadata=metadata).serve_forever()


if __name__=='__main__':main()
