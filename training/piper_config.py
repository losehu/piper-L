"""Single real PiPER: 6 joints [rad] + gripper opening [m], two RGB views."""
import dataclasses,os,pathlib
import numpy as np
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies.libero_policy import _parse_image
from openpi.training import config,optimizer,weight_loaders

ROOT=pathlib.Path(os.environ.get('PIPER_RUN_ROOT',str(pathlib.Path(__file__).resolve().parents[1]/'artifacts/training'))).expanduser().resolve()
BASE_PARAMS=os.environ.get('PIPER_BASE_PARAMS',str(pathlib.Path(os.environ.get('OPENPI_DATA_HOME',str(ROOT/'cache/openpi')))/'openpi-assets/checkpoints/pi05_base/params'))
DATA_ROOT=pathlib.Path(os.environ.get('HF_LEROBOT_HOME',str(ROOT/'lerobot'))).expanduser().resolve()
NAME='pi05_piper_grasp'
PROMPT='Pick up all the objects on the table and place them in the box.'

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
 def __call__(self,data):
  base=_parse_image(data['cam0']);wrist=_parse_image(data['cam1'])
  state=np.asarray(data['state'],dtype=np.float32)
  if state.shape!=(7,):raise ValueError(f'Expected 7D measured PiPER state, got {state.shape}')
  result={'state':state,'image':{'base_0_rgb':base,'left_wrist_0_rgb':wrist,'right_wrist_0_rgb':np.zeros_like(base)},'image_mask':{'base_0_rgb':np.True_,'left_wrist_0_rgb':np.True_,'right_wrist_0_rgb':np.False_},'prompt':data.get('prompt',PROMPT)}
  if 'actions' in data:result['actions']=np.array(data['actions'],dtype=np.float32,copy=True)
  return result

@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
 def __call__(self,data):return {'actions':np.asarray(data['actions'])[...,:7]}

@dataclasses.dataclass(frozen=True)
class PiperDataConfig(config.DataConfigFactory):
 def create(self,assets_dirs,model_config):
  mask=transforms.make_bool_mask(6,-1)
  group=transforms.Group(inputs=[PiperInputs(),transforms.DeltaActions(mask)],outputs=[transforms.AbsoluteActions(mask),PiperOutputs()])
  return dataclasses.replace(self.create_base_config(assets_dirs,model_config),repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({'cam0':'observation.images.cam0','cam1':'observation.images.cam1','state':'observation.state','actions':'action','prompt':'prompt'})]),data_transforms=group,model_transforms=config.ModelTransformFactory(default_prompt=PROMPT)(model_config),action_sequence_keys=('action',))

def get_config(partition='train',**overrides):
 model=pi0_config.Pi0Config(pi05=True,action_horizon=50)
 result=config.TrainConfig(name=NAME,project_name='piper_real_grasp',exp_name='screened54_train48_v1',model=model,data=PiperDataConfig(repo_id='local/piper_grasp_'+partition,assets=config.AssetsConfig(assets_dir=str(ROOT/'assets'/NAME),asset_id='local/piper_grasp_train'),base_config=config.DataConfig(prompt_from_task=True)),weight_loader=weight_loaders.CheckpointWeightLoader(BASE_PARAMS),lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=50,peak_lr=1e-5,decay_steps=2500,decay_lr=1e-6),optimizer=optimizer.AdamW(clip_gradient_norm=1.0),ema_decay=.999,batch_size=64,num_workers=12,num_train_steps=2500,log_interval=10,save_interval=125,keep_period=250,assets_base_dir=str(ROOT/'assets'),checkpoint_base_dir=str(ROOT/'checkpoints'),wandb_enabled=False,fsdp_devices=2,policy_metadata={'robot':'piper_single_follower','fps':30,'control_hz':120,'action_semantics':'absolute joint targets [rad] and gripper opening [m] after inverse delta transform','physical_action_dim':7,'cam0':'third_person_rgb','cam1':'wrist_rgb','replan_frames':10,'task':PROMPT,'training_source':'screened54, 48 train / 6 held-out val'})
 return dataclasses.replace(result,**overrides)

config._CONFIGS_DICT[NAME]=get_config()
