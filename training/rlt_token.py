"""Train the unmodified upstream Token encoder/decoder on eight H100s."""
import argparse,dataclasses,json,time
from pathlib import Path
import numpy as np
from rlt_piper.contract import upstream,SCHEMA,UPSTREAM,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--devices',type=int,default=8);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];upstream(root/'deployment')
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from rl_token import RLTokenConfig,RLTokenModel
    devices=jax.local_devices();assert len(devices)==a.devices
    cfg=RLTokenConfig();model=RLTokenModel(cfg);rng=np.random.default_rng(42)
    data={}
    for split in ['train','val']:
        arrays=[np.load(d/'prefix.npy',mmap_mode='r') for d in sorted(a.features.glob(split+'_*')) if (d/'complete.json').exists()]
        if not arrays:raise RuntimeError('Missing split: '+split)
        data[split]=(arrays,[(i,j) for i,x in enumerate(arrays) for j in range(len(x))])
    def batch(split):
        arrays,index=data[split];rows=[index[k] for k in rng.integers(0,len(index),size=a.devices)]
        return np.stack([arrays[i][j] for i,j in rows]).astype(np.float32)[:,None]
    initial=model.init(jax.random.key(0),jnp.zeros((1,512,2048)),train=False)
    tx=optax.adamw(optax.cosine_decay_schedule(3e-4,a.steps,alpha=.05),weight_decay=1e-4)
    params=jax.device_put_replicated(initial,devices);optim=jax.device_put_replicated(tx.init(initial),devices)
    def step(params,optim,x):
        def loss(p):return model.apply(p,x,train=False,method=model.loss)[0]
        value,grad=jax.value_and_grad(loss)(params);grad=jax.lax.pmean(grad,'gpu')
        update,optim=tx.update(grad,optim,params)
        return optax.apply_updates(params,update),optim,jax.lax.pmean(value,'gpu')
    train=jax.pmap(step,axis_name='gpu');evaluate=jax.pmap(lambda p,x:model.apply(p,x,train=False,method=model.loss)[0])
    a.output.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');initial_val=None
    for i in range(a.steps+1):
        if i%100==0 or i==a.steps:
            val=float(np.mean([np.asarray(evaluate(params,batch('val'))).mean() for _ in range(8)]))
            if initial_val is None:initial_val=val
            row=dict(step=i,val_mse=val);history.append(row);print(json.dumps(row),flush=True)
            if val<best:
                best=val;host=jax.tree.map(lambda x:np.asarray(x[0]),params)
                temp=a.output/'token.msgpack.partial';temp.write_bytes(serialization.to_bytes(host));temp.replace(a.output/'token.msgpack')
            (a.output/'token_history.json').write_text(json.dumps(history,indent=2)+'\n')
        if i==a.steps:break
        params,optim,loss=train(params,optim,batch('train'))
    if not np.isfinite(best) or best>=initial_val:raise RuntimeError('Token validation did not improve; no ready metadata written')
    meta=dict(schema=SCHEMA,upstream=UPSTREAM,base_step=2500,base_frozen=True,trained=True,encoder=dataclasses.asdict(cfg),
              input_shape=[512,2048],z_dim=512,steps=a.steps,devices=a.devices,val_mse=best,initial_val_mse=initial_val,
              token_sha256=digest(a.output/'token.msgpack'),norm_sha256=digest(a.output/'norm_stats.json'),
              features='post-transformer two valid camera prefixes; permanently masked third camera excluded')
    (a.output/'token.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()
