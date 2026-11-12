"""Use PyAV for this dataset; server TorchCodec lacks system FFmpeg libraries."""
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.training import data_loader
from openpi import transforms

def install():
 original=data_loader.create_torch_dataset
 def create(data_config,action_horizon,model_config):
  if not (data_config.repo_id or '').startswith('local/piper_grasp_'):
   return original(data_config,action_horizon,model_config)
  meta=lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
  dataset=lerobot_dataset.LeRobotDataset(data_config.repo_id,delta_timestamps={key:[i/meta.fps for i in range(action_horizon)] for key in data_config.action_sequence_keys},video_backend='pyav')
  if data_config.prompt_from_task:dataset=data_loader.TransformedDataset(dataset,[transforms.PromptFromLeRobotTask(meta.tasks)])
  return dataset
 data_loader.create_torch_dataset=create
