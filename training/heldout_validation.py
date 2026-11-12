"""Fixed held-out flow loss; not a measurement of real-robot success."""
import dataclasses,json,logging,pathlib
import flax.nnx as nnx
import jax,jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from openpi.models import model as model_lib
from openpi.training import data_loader,sharding
import piper_config

def install(train,cfg):
 valcfg=piper_config.get_config('val');dc=valcfg.data.create(valcfg.assets_dirs,valcfg.model)
 dataset=data_loader.create_torch_dataset(dc,50,valcfg.model);dataset=data_loader.transform_dataset(dataset,dc)
 samples=[];offset=0
 for p in sorted((piper_config.DATA_ROOT/'local/piper_grasp_val/data').rglob('*.parquet')):
  n=pq.read_metadata(p).num_rows
  for i in np.linspace(0,max(0,n-50),8).astype(int):samples.append(dataset[offset+int(i)])
  offset+=n
 batches=[]
 for start in range(0,len(samples),8):
  batch=jax.tree.map(lambda *x:np.stack(x),*samples[start:start+8]);batches.append((model_lib.Observation.from_dict(batch),batch['actions']))
 original=train._checkpoints.save_state;compiled=None;best=float('inf')
 def evaluate(state,step):
  nonlocal compiled,best
  if compiled is None:
   definition=state.model_def
   @jax.jit
   def val_step(params,obs,actions,rng):
    model=nnx.merge(definition,params);model.eval()
    return jnp.mean(model.compute_loss(rng,obs,actions,train=False))
   compiled=val_step
  params=state.ema_params if state.ema_params is not None else state.params
  with sharding.set_mesh(sharding.make_mesh(cfg.fsdp_devices)):
   values=[float(compiled(params,obs,actions,jax.random.key(20261111+i))) for i,(obs,actions) in enumerate(batches)]
  value=float(np.mean(values));record=dict(step=int(step),heldout_flow_loss=value,batches=values,val_episodes=6,val_frames_sampled=len(samples),uses_ema=state.ema_params is not None,real_robot_success_evaluated=False)
  with (piper_config.ROOT/'logs/validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  logging.info('HELDOUT step=%d flow_loss=%.6f (not real-robot success)',step,value)
  if np.isfinite(value) and value<best and (step % cfg.keep_period == 0 or step == cfg.num_train_steps-1):
   best=value;(piper_config.ROOT/'best_checkpoint.json').write_text(json.dumps(dict(record,checkpoint=str(cfg.checkpoint_dir/str(step))),indent=2)+'\n')
 def save_and_validate(manager,state,loader,step):
  original(manager,state,loader,step)
  evaluate(state,step)
 train._checkpoints.save_state=save_and_validate
 original_init=train.init_train_state
 def init_and_validate(*args,**kwargs):
  result=original_init(*args,**kwargs)
  if not kwargs.get('resume',False):evaluate(result[0],-1)
  return result
 train.init_train_state=init_and_validate
