"""The trained checkpoint's transforms: six delta joints, absolute gripper."""
import dataclasses
from pathlib import Path

import numpy as np
from openpi import transforms
from openpi.models import pi0_config
from openpi.policies.libero_policy import _parse_image
from openpi.training import config

PROMPT = 'Pick up all the objects on the table and place them in the box.'


@dataclasses.dataclass(frozen=True)
class Inputs(transforms.DataTransformFn):
    def __call__(self, data):
        base, wrist = _parse_image(data['cam0']), _parse_image(data['cam1'])
        state = np.asarray(data['state'], dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('Expected seven finite radians/metres state values')
        out = dict(state=state,
                   image=dict(base_0_rgb=base, left_wrist_0_rgb=wrist,
                              right_wrist_0_rgb=np.zeros_like(base)),
                   image_mask=dict(base_0_rgb=np.True_, left_wrist_0_rgb=np.True_,
                                   right_wrist_0_rgb=np.False_),
                   prompt=data.get('prompt', PROMPT))
        if 'actions' in data:
            out['actions'] = np.array(data['actions'], dtype=np.float32, copy=True)
        return out


@dataclasses.dataclass(frozen=True)
class Outputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {'actions': np.asarray(data['actions'])[..., :7]}


@dataclasses.dataclass(frozen=True)
class Data(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        mask = transforms.make_bool_mask(6, -1)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=transforms.Group(
                inputs=[Inputs(), transforms.DeltaActions(mask)],
                outputs=[transforms.AbsoluteActions(mask), Outputs()]),
            model_transforms=config.ModelTransformFactory(default_prompt=PROMPT)(model_config),
            action_sequence_keys=('action',))


def training_config(root):
    root = Path(root)
    return config.TrainConfig(
        name='pi05_piper_grasp', exp_name='screened54_train48_v1',
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=Data(repo_id='local/piper_grasp_train',
                  assets=config.AssetsConfig(asset_id='local/piper_grasp_train'),
                  base_config=config.DataConfig(prompt_from_task=True)),
        assets_base_dir=str(root / 'assets'), checkpoint_base_dir=str(root / 'checkpoints'),
        batch_size=1, num_workers=0, num_train_steps=0, wandb_enabled=False)
