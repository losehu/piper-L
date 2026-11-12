"""OpenPI's native websocket server; saved-observation warmup, no hardware imports."""
import argparse
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    root = args.root.resolve()
    state_path = root / 'logs/policy_state.json'
    state = dict(pid=os.getpid(), status='loading', hardware_opened=False)

    def write():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(state_path)

    write()
    import jax
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from rtc_piper.model_adapter import training_config, PROMPT
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Local GPU required; CPU fallback is refused')
    settings=yaml.safe_load((root/'config.yaml').read_text())
    if (settings['model_step']!=2500 or settings['checkpoint_index']!=2499
            or settings['checkpoint']!='checkpoints/pi05_piper_2500_bf16'):
        raise RuntimeError('Expected requested final2500-update checkpoint')
    checkpoint = root / settings['checkpoint']
    policy = policy_config.create_trained_policy(
        training_config(root), checkpoint, default_prompt=PROMPT,
        sample_kwargs={'num_steps': 10})

    class CheckedPolicy:
        failed = False
        def infer(self, obs):
            if self.failed:
                raise RuntimeError('Policy error latched; restart policy service')
            try:
                request_id = int(obs.pop('request_id', 0))
                observed_at = float(obs.pop('observed_at', 0))
                values = np.asarray(obs['state'], dtype=np.float32)
                if values.shape != (7,) or not np.isfinite(values).all():
                    raise ValueError('Invalid 7D state')
                result = policy.infer(obs)
                actions = np.asarray(result['actions'], dtype=np.float32)
                if actions.shape != (50, 7) or not np.isfinite(actions).all():
                    raise ValueError('Expected finite 50x7 actions')
                return dict(actions=actions, request_id=request_id, observed_at=observed_at)
            except Exception as error:
                self.failed = True
                state.update(status='failed', error=str(error))
                write()
                raise

    checked = CheckedPolicy()
    with np.load(root / 'offline_sample.npz', allow_pickle=False) as sample:
        observation = {key:np.array(sample[key], copy=True) for key in ['cam0','cam1','state']}
        observation['prompt'] = str(sample['prompt'].item())
    timings = []
    for _ in range(4):
        before = time.monotonic()
        checked.infer({key:value.copy() if isinstance(value, np.ndarray) else value
                       for key,value in observation.items()})
        timings.append(time.monotonic()-before)
    metadata = dict(model='pi05_piper_grasp_%d'%settings['model_step'], checkpoint=str(checkpoint),
                    training_updates=settings['model_step'],source_checkpoint_index=settings['checkpoint_index'],
                    action_horizon=50, physical_action_dim=7, fps=30,
                    action_semantics='absolute joint radians and gripper metres',
                    robot='piper_l', backend=jax.default_backend(),
                    gpu=[d.device_kind for d in jax.devices()],
                    warmed_up=True, warm_p95_s=float(np.percentile(timings[1:],95)))
    state.update(status='ready', metadata=metadata, warmup_seconds=timings)
    write()
    logging.info('Ready: %s', metadata)
    WebsocketPolicyServer(checked, host='127.0.0.1', port=args.port,
                          metadata=metadata).serve_forever()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
