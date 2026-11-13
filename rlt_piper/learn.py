"""Between-episode updates with the unmodified upstream actor/critic trainer."""
import argparse,dataclasses,fcntl,json,os,pickle,time
from pathlib import Path
import numpy as np
from rlt_piper.contract import assets,upstream,normalize,SCHEMA,CHUNK,digest
from rlt_piper.recording import transitions
from rlt_piper.checkpoint import current,publish

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2000);a=p.parse_args();meta=assets(a.assets);upstream(a.root)
    import jax,jax.numpy as jnp
    from flax import serialization
    from rlt_online_rl.config import RLTOnlineRLConfig
    from rlt_online_rl.trainer import init_train_state,train_step
    lock=(a.assets/'learner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    episodes=[];rows=[];settings=None
    for path in sorted((a.root/'rlt_episodes').glob('episode_*')):
        if not (path/'episode.json').is_file():continue
        found=transitions(path,meta['token_sha256'])
        if not found:continue
        m=json.loads((path/'episode.json').read_text())
        if settings is None:settings=m['settings']
        if m['settings']!=settings:raise RuntimeError('RTC/filter configuration changed across replay episodes')
        rows+=found;episodes.append(path.name)
    # Match the public Ethernet recipe's replay warmup size, without inventing rewards.
    if len(rows)<600:
        print(json.dumps(dict(status='waiting_for_labelled_rollouts',transitions=len(rows),required=600)),flush=True);return
    norms=json.loads((a.assets/'norm_stats.json').read_text());norms=norms.get('norm_stats',norms)
    cfg=RLTOnlineRLConfig(z_dim=meta['z_dim'],chunk_len=CHUNK,action_representation='delta_chunk',
                          warmup_min_size=600,warmup_post_collect_updates=2000,fixed_std=.05)
    # Upstream state serialization expects the portable uint32 PRNG key format.
    state,actor,critic=init_train_state(cfg,rng=jax.random.PRNGKey(42));previous=[]
    checkpoint=current(a.assets)
    if (checkpoint/'learner.msgpack').exists():
        prior=json.loads((checkpoint/'actor.json').read_text())
        if prior['token_sha256']!=meta['token_sha256']:raise RuntimeError('Learner/token lineage changed')
        state=serialization.from_bytes(state,(checkpoint/'learner.msgpack').read_bytes());previous=prior['episodes']
        if prior['execution_settings']!=settings:raise RuntimeError('Learner execution configuration changed')
    if set(episodes)<=set(previous):
        print(json.dumps(dict(status='no_new_labelled_episodes',episodes=len(episodes))),flush=True);return
    keys=['z_rl','proprio','ref_chunk','action_chunk','rewards','done','next_z_rl','next_proprio','next_ref_chunk','source_chunk']
    data={k:np.stack([row[k] for row in rows]) for k in keys}
    data['ref_chunk']=normalize(data['ref_chunk'],data['proprio'],norms['actions'])
    data['action_chunk']=normalize(data['action_chunk'],data['proprio'],norms['actions'])
    data['next_ref_chunk']=normalize(data['next_ref_chunk'],data['next_proprio'],norms['actions'])
    rng=np.random.default_rng(43);metrics=None
    for i in range(a.steps):
        ix=rng.integers(0,len(rows),size=256);batch={k:jnp.asarray(v[ix]) for k,v in data.items()}
        q_weight=0. if int(state.global_step)<2000 else .1
        state,metrics=train_step(state,batch,actor=actor,critic=critic,rl_config=cfg,bc_weight=5.,q_weight=q_weight,delta_weight=0.)
        if i%100==0:
            report={k:float(v) for k,v in metrics.items()}
            if not all(np.isfinite(v) for v in report.values()):raise RuntimeError('Nonfinite RLT training metrics')
            print(json.dumps(dict(step=int(state.global_step),transitions=len(rows),**report)),flush=True)
    if not all(np.isfinite(float(v)) for v in metrics.values()):raise RuntimeError('Nonfinite final RLT metrics')
    state=jax.tree.map(lambda x:np.asarray(x),state);version=int(state.actor_version)
    payload=dict(version=version,actor_params=state.actor_params,rl_config=dataclasses.asdict(cfg),token_sha256=meta['token_sha256'])
    info=dict(schema=SCHEMA,token_sha256=meta['token_sha256'],chunk_len=CHUNK,actor_updates=version,
              global_step=int(state.global_step),episodes=episodes,transitions=len(rows),execution_settings=settings,
              ready=int(state.global_step)>=2000,metrics={k:float(v) for k,v in metrics.items()},
              stage='bc_warmup' if int(state.global_step)<=2000 else 'online_rl',time=time.time(),real_robot_evaluation='not yet validated')
    publish(a.assets,serialization.to_bytes(state),payload,info)
    print(json.dumps(dict(status='actor_updated',version=version,stage=info['stage'])),flush=True)

if __name__=='__main__':main()
