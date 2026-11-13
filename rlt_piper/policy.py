"""Pinned upstream Token + ChunkActor around the frozen existing pi05 policy."""
import dataclasses
import json
import pickle
import numpy as np
from rlt_piper.contract import assets,upstream,normalize,absolute,CHUNK

class Policy:
    def __init__(self,base,root,directory,stage):
        import jax
        import jax.numpy as jnp
        from flax import serialization
        from rlt_piper.features import Extractor
        upstream(root)
        from rl_token import RLTokenModel,RLTokenConfig
        from rlt_online_rl.config import RLTOnlineRLConfig
        from rlt_online_rl.networks import ChunkActor
        self.base,self.stage=base,stage;self.meta=assets(directory,stage=='eval')
        self.norms=json.loads((directory/'norm_stats.json').read_text());self.norms=self.norms.get('norm_stats',self.norms)
        self.extract=Extractor(base)
        self.token=RLTokenModel(RLTokenConfig(**self.meta['encoder']))
        self.params=serialization.msgpack_restore((directory/'token.msgpack').read_bytes())
        self.encode_token=jax.jit(lambda x:self.token.apply(self.params,x,train=False,method=self.token.encode).reshape(x.shape[0],-1))
        self.last=None;self.version=0;self.actor=None;self.pending=None;self.directory=directory
        cfg=RLTOnlineRLConfig(z_dim=self.meta['z_dim'],chunk_len=CHUNK,action_representation='delta_chunk')
        self.network=ChunkActor(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.actor_hidden_dim,cfg.actor_num_layers,cfg.fixed_std)
        self.actor_mean=jax.jit(lambda p,z,s,a:self.network.actor_mean(p,z,s,a))
        self.actor_sample=jax.jit(lambda p,key,z,s,a:self.network.sample_action(p,key,z,s,a,deterministic=False))
        self.rng=jax.random.PRNGKey(43)
        # Compile actor inference before hardware preparation, including first activation.
        params=self.network.init_params(jax.random.PRNGKey(7))
        args=(jnp.zeros((1,cfg.z_dim)),jnp.zeros((1,7)),jnp.zeros((1,CHUNK,7)))
        jax.block_until_ready(self.actor_mean(params,*args))
        jax.block_until_ready(self.actor_sample(params,self.rng,*args))
        if stage!='collect':
            from rlt_piper.checkpoint import snapshot
            found=snapshot(directory,self.meta['token_sha256'])
            if found:self.install(*found)

    def install(self,payload,meta):
        import jax,jax.numpy as jnp
        cfg=payload['rl_config']
        expected=dict(z_dim=self.meta['z_dim'],chunk_len=CHUNK,action_dim=7,proprio_dim=7,
                      action_representation='delta_chunk',actor_hidden_dim=256,actor_num_layers=2,fixed_std=.05)
        if any(cfg.get(k)!=v for k,v in expected.items()):raise RuntimeError('RLT actor architecture/action identity mismatch')
        self.actor_params=jax.tree.map(jnp.asarray,payload['actor_params'])
        self.actor=self.network;self.version=int(payload['version'])

    def features(self,obs):
        return np.asarray(self.encode_token(self.extract(obs)[None,...])[0])

    def infer(self,obs,features_only=False):
        import jax
        import jax.numpy as jnp
        # Pending parameters activate once, before producing a complete new chunk.
        if self.pending is not None and not features_only:
            self.install(*self.pending);self.pending=None
        result=self.base.infer(obs)['actions'];ref=np.array(result,copy=True)
        token=self.features(obs);source='base'
        if self.actor is not None and not features_only:
            a=normalize(ref[:CHUNK],obs['state'],self.norms['actions'])
            args=(jnp.asarray(token)[None],jnp.asarray(obs['state'])[None],jnp.asarray(a)[None])
            if self.stage=='online':
                self.rng,key=jax.random.split(self.rng);out=np.asarray(self.actor_sample(self.actor_params,key,*args)[0])
            else:out=np.asarray(self.actor_mean(self.actor_params,*args)[0])
            result=np.array(result,copy=True);result[:CHUNK]=absolute(out,obs['state'],self.norms['actions']);source='rlt'
        self.last=dict(z_rl=token,proprio=np.array(obs['state']),ref_chunk=ref[:CHUNK],
                       proposed_chunk=np.array(result[:CHUNK]),source=source,actor_version=self.version)
        return result
