"""Synthetic single-GPU gradient/serialization smoke; never writes live actors."""
import argparse,json
from pathlib import Path
import numpy as np
from rlt_piper.contract import upstream

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    upstream(a.root)
    import jax,jax.numpy as jnp
    from flax import serialization
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.trainer import init_train_state,train_step
    if jax.default_backend()!='gpu':raise RuntimeError('This validation must run on the5090')
    cfg=RLTOnlineRLConfig(z_dim=512,chunk_len=18);state,actor,critic=init_train_state(cfg,rng=jax.random.PRNGKey(0));initial=state
    rng=np.random.default_rng(1);b=4
    batch=dict(z_rl=jnp.asarray(rng.normal(size=(b,512)),jnp.float32),proprio=jnp.zeros((b,7)),ref_chunk=jnp.zeros((b,18,7)),
               action_chunk=jnp.zeros((b,18,7)),rewards=jnp.ones((b,18))*.01,done=jnp.ones(b),
               next_z_rl=jnp.zeros((b,512)),next_proprio=jnp.zeros((b,7)),next_ref_chunk=jnp.zeros((b,18,7)),source_chunk=jnp.zeros((b,18),jnp.int32))
    for _ in range(4):state,metrics=train_step(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=.1)
    if not all(np.isfinite(float(v)) for v in metrics.values()):raise RuntimeError('Nonfinite synthetic learner metrics')
    if not any(np.any(np.asarray(x)!=np.asarray(y)) for x,y in zip(jax.tree.leaves(initial.actor_params),jax.tree.leaves(state.actor_params))):raise RuntimeError('Actor was not updated')
    loaded=serialization.from_bytes(initial,serialization.to_bytes(state))
    if int(loaded.global_step)!=4 or int(loaded.actor_version)!=2:raise RuntimeError('Learner state roundtrip failed')
    mean=actor.actor_mean(loaded.actor_params,batch['z_rl'],batch['proprio'],batch['ref_chunk'])
    if mean.shape!=(b,18,7) or not np.isfinite(np.asarray(mean)).all():raise RuntimeError('Bad actor inference')
    report=dict(status='passed',hardware_opened=False,robot_commands_sent=0,synthetic=True,live_actor_written=False,
                backend=jax.default_backend(),gradient_steps=4,actor_updates=2,metrics={k:float(v) for k,v in metrics.items()})
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
