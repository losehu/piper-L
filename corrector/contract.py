"""Identity and action conventions shared by training and deployment."""
import hashlib
import json
import numpy as np

VERSION='piper-pi05-corrector-v1'
CAMERAS=('base_0_rgb','left_wrist_0_rgb')


def action_normalize(action,state,stats):
    action=np.asarray(action,dtype=np.float32).copy()
    action[...,:6]-=np.asarray(state,dtype=np.float32)[...,:6]
    q01=np.asarray(stats['q01'],dtype=np.float32)[:7]
    q99=np.asarray(stats['q99'],dtype=np.float32)[:7]
    # Identical to OpenPI Normalize(use_quantiles=True), without extra clipping.
    return (action-q01)/(q99-q01+1e-6)*2-1


def identity(norm_stats,k_step=5):
    return dict(version=VERSION,training_updates=2500,checkpoint_index=2499,
        camera_order=list(CAMERAS),fps=30,action_dim=7,k_step=int(k_step),
        action_semantics='OpenPI normalized six joint deltas and absolute gripper',
        norm_sha256=hashlib.sha256(json.dumps(norm_stats,sort_keys=True).encode()).hexdigest())
