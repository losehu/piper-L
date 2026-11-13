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
        self.base,self.stage=base,stage;self.meta=assets(directory,stage!='collect')
        self.norms=json.loads((directory/'norm_stats.json').read_text());self.norms=self.norms.get('norm_stats',self.norms)
        self.extract=Extractor(base)
        self.token=RLTokenModel(RLTokenConfig(**self.meta['encoder']))
        self.params=serialization.msgpack_restore((directory/'token.msgpack').read_bytes())
        self.encode_token=jax.jit(lambda x:self.token.apply(self.params,x,train=False,method=self.token.encode).reshape(x.shape[0],-1))
        self.version=0;self.actor=None
        if stage!='collect':
            with (directory/'actor.pkl').open('rb') as f:snapshot=pickle.load(f)
            cfg=RLTOnlineRLConfig(**snapshot['rl_config'])
            if (cfg.z_dim!=self.meta['z_dim'] or cfg.chunk_len!=CHUNK or cfg.action_dim!=7 or cfg.proprio_dim!=7
                    or cfg.action_representation!='delta_chunk' or int(snapshot['version'])<1):
                raise RuntimeError('RLT actor architecture/action identity mismatch')
            self.actor=ChunkActor(cfg.z_dim,cfg.proprio_dim,cfg.chunk_len,cfg.action_dim,cfg.actor_hidden_dim,cfg.actor_num_layers,cfg.fixed_std)
            self.actor_params=jax.tree.map(jnp.asarray,snapshot['actor_params'])
            self.actor_mean=jax.jit(lambda z,s,a:self.actor.actor_mean(self.actor_params,z,s,a))
            self.version=int(snapshot['version']);self.rng=jax.random.key(43)
            self.actor_sample=jax.jit(lambda key,z,s,a:self.actor.sample_action(self.actor_params,key,z,s,a,deterministic=False))
        self.last=None

    def features(self,obs):
        return np.asarray(self.encode_token(self.extract(obs)[None,...])[0])

    def infer(self,obs):
        import jax
        import jax.numpy as jnp
        result=self.base.infer(obs)['actions'];ref=np.array(result,copy=True)
        token=self.features(obs);source='base'
        if self.actor is not None:
            a=normalize(ref[:CHUNK],obs['state'],self.norms['actions'])
            args=(jnp.asarray(token)[None],jnp.asarray(obs['state'])[None],jnp.asarray(a)[None])
            if self.stage=='online':
                self.rng,key=jax.random.split(self.rng);out=np.asarray(self.actor_sample(key,*args)[0])
            else:out=np.asarray(self.actor_mean(*args)[0])
            result=np.array(result,copy=True);result[:CHUNK]=absolute(out,obs['state'],self.norms['actions']);source='rlt'
        self.last=dict(z_rl=token,proprio=np.array(obs['state']),ref_chunk=ref[:CHUNK],
                       proposed_chunk=np.array(result[:CHUNK]),source=source,actor_version=self.version)
        return result
