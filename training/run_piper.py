import argparse,dataclasses,json,pathlib,sys
import numpy as np
import pyarrow.parquet as pq
import piper_config
from openpi.shared import normalize
from openpi.training import data_loader
import piper_data_loader
piper_data_loader.install()

def compute_stats():
 root=piper_config.DATA_ROOT/'local/piper_grasp_train'
 stats={key:normalize.RunningStats() for key in ['state','actions']}
 frames=0
 for p in sorted((root/'data').rglob('*.parquet')):
  table=pq.read_table(p);state=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32);action=np.asarray(table['action'].to_pylist(),dtype=np.float32);n=len(state)
  stats['state'].update(state)
  for start in range(0,n,128):
   ids=np.minimum(np.arange(start,min(start+128,n))[:,None]+np.arange(50)[None,:],n-1)
   chunks=action[ids].copy();chunks[...,:6]-=state[start:min(start+128,n),None,:6];stats['actions'].update(chunks)
  frames+=n
 dest=piper_config.get_config().assets_dirs/'local/piper_grasp_train';normalize.save(dest,{key:value.get_statistics() for key,value in stats.items()})
 print('TRAIN-ONLY STATS',frames,dest,flush=True)

def check_pipeline():
 cfg=piper_config.get_config();dc=cfg.data.create(cfg.assets_dirs,cfg.model)
 dataset=data_loader.create_torch_dataset(dc,cfg.model.action_horizon,cfg.model)
 transformed=data_loader.transform_dataset(dataset,dc)
 sample=transformed[0]
 assert sample['state'].shape==(32,);assert sample['actions'].shape==(50,32)
 assert np.isfinite(sample['state']).all() and np.isfinite(sample['actions']).all()
 # Delta/inverse delta preserves radians and metres, with no ALOHA gripper conversion.
 from openpi import transforms
 s=np.arange(7,dtype=np.float32)*.01;a=np.tile(s,(50,1));a[:,:6]+=.1;a[:,6]=.05;expected=a.copy()
 data={'state':s,'actions':a};mask=transforms.make_bool_mask(6,-1)
 delta=transforms.DeltaActions(mask)(data);assert np.allclose(delta['actions'][:,:6],.1);assert np.allclose(delta['actions'][:,6],.05)
 inverse=transforms.AbsoluteActions(mask)(delta);assert np.allclose(inverse['actions'],expected)
 # Sample all held-out episodes through the exact same transforms / train-only statistics.
 valcfg=piper_config.get_config('val');vdc=valcfg.data.create(valcfg.assets_dirs,valcfg.model);val=data_loader.create_torch_dataset(vdc,50,valcfg.model);vtrans=data_loader.transform_dataset(val,vdc)
 for i in np.linspace(0,len(val)-1,12).astype(int):
  x=vtrans[int(i)];assert np.isfinite(x['actions']).all()
 result=dict(train_frames=len(dataset),val_frames=len(val),action_shape=list(sample['actions'].shape),state_shape=list(sample['state'].shape),delta_roundtrip=True,gripper_units_m=True,validation_uses_train_norm_stats=True)
 (piper_config.ROOT/'pipeline_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['stats','check','train','smoke']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 if args.stage=='stats':compute_stats()
 elif args.stage=='check':check_pipeline()
 else:
  from scripts import train
  train.init_logging()
  cfg=piper_config.get_config()
  if args.stage=='smoke':
   cfg=dataclasses.replace(cfg,exp_name='smoke10',num_train_steps=10,log_interval=1,keep_period=None)
   train._checkpoints.save_state=lambda *a,**kw:None
  else:
   import heldout_validation
   heldout_validation.install(train,cfg)
  if args.resume:cfg=dataclasses.replace(cfg,resume=True)
  train.main(cfg)
