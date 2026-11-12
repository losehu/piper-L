"""Export the trained small MLP and verify PyTorch/JAX value+gradient parity."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--norm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    import jax
    import jax.numpy as jnp
    from corrector.vendor.vla_corrector.MLP import SiglipResidualMLP
    from corrector.vendor.vla_corrector.config import SiglipMLPConfig
    from corrector.jax_mlp import forward
    torch.set_num_threads(2);payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['model_config'];model=SiglipResidualMLP(SiglipMLPConfig(token_dim=cfg['token_dim'],action_dim=7));model.load_state_dict(payload['state_dict']);model.eval()
    rng=np.random.default_rng(42);z=rng.normal(0,.4,(1,8,cfg['token_dim'])).astype(np.float32);a=rng.normal(0,.2,(1,7)).astype(np.float32)
    weights={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    expected=model(torch.from_numpy(z),torch.from_numpy(a)).detach().numpy()
    actual=np.asarray(forward({k:jnp.asarray(v) for k,v in weights.items()},jnp.asarray(z),jnp.asarray(a)))
    error=float(np.max(abs(expected-actual)))
    action=torch.tensor(a,requires_grad=True);model(torch.from_numpy(z),action).square().mean().backward();tg=action.grad.numpy()
    jg=np.asarray(jax.grad(lambda action:jnp.mean(forward(weights,jnp.asarray(z),action)**2))(jnp.asarray(a)))
    grad_error=float(np.max(abs(tg-jg)))
    if error>2e-4 or grad_error>2e-4:raise RuntimeError('PyTorch/JAX parity failed: '+str((error,grad_error)))
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez(args.output/'mlp.npz',**weights);shutil.copy2(args.norm,args.output/'norm_stats.json')
    scores=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(args.checkpoint.parent.glob('best_scores_rank*.npy'))])
    median=float(np.median(scores));mad=float(np.median(abs(scores-median)))
    monitor_config=dict(window=100,warmup=20,persistence=3,cooldown_s=.5,
        on_mad=4.,off_mad=2.,floor=float(np.quantile(scores,.95)),mad_floor=.005,max_interpolation_gap_s=.2,
        calibration='six held-out episodes used for model selection and normal-score calibration; not an independent physical test',
        calibration_median=median,calibration_mad=mad,calibration_p99=float(np.quantile(scores,.99)))
    metadata=dict(contract=payload['contract'],model_config=cfg,parameters=payload['parameters'],epoch=payload['epoch'],metrics=payload['metrics'],monitor_config=monitor_config,
        torch_jax_max_abs=error,torch_jax_gradient_max_abs=grad_error,
        feature_tokens=512,hidden_dtype='float32',provenance=dict(repository='https://github.com/ZJU-OmniAI/vla-corrector',commit='927a06da8f00d7df483459e9039219e076c846de'),
        physical_robot_tested=False)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
