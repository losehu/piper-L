"""Frozen OpenPI PI0.5 with optional small-MLP velocity guidance and latent reuse.

The sampler mirrors OpenPI Pi0.sample_actions; original vendor code is untouched.
The monitor receives the same image tokens already used by policy inference.
Sampler adapted from the vendored OpenPI implementation (Apache-2.0);
see deployment/vendor/openpi/LICENSE and docs/third_party.md.
"""
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import einops
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask
from openpi.shared.nnx_utils import module_jit
from corrector.jax_mlp import forward,guide_velocity


class CorrectedModel(nnx.Module):
    def __init__(self,base,weights,guidance_every=5):
        self.base=base;self.weights=nnx.Dict({k:nnx.Param(v) for k,v in weights.items()})
        self.guidance_every=guidance_every

    def _weights(self):return {k:v.value for k,v in self.weights.items()}

    def predict(self,z,action):return forward(self._weights(),z,action)

    def encode(self,images):
        tokens,_=self.base.PaliGemma.img(images,train=False)
        return jnp.concatenate([tokens[:1],tokens[1:2]],axis=1).astype(jnp.float32)

    def sample(self,rng,observation,noise,target,enabled,eta):
        base=self.base;observation=model_module.preprocess_observation(None,observation,train=False)
        batch_size=observation.state.shape[0];num_steps=10;dt=-1./num_steps
        prefix,prefix_mask,prefix_ar=base.embed_prefix(observation)
        image_tokens=(prefix.shape[1]-observation.tokenized_prompt.shape[1])//3
        z=prefix[:,:2*image_tokens,:].astype(jnp.float32)
        prefix_attn=make_attn_mask(prefix_mask,prefix_ar);positions=jnp.cumsum(prefix_mask,axis=1)-1
        _,cache=base.PaliGemma.llm([prefix,None],mask=prefix_attn,positions=positions)
        weights=self._weights()
        def step(carry):
            x_t,time,index=carry
            suffix,mask,ar,ad=base.embed_suffix(observation,x_t,jnp.broadcast_to(time,batch_size))
            suffix_attn=make_attn_mask(mask,ar)
            prefix_attn=einops.repeat(prefix_mask,'b p -> b s p',s=suffix.shape[1])
            attention=jnp.concatenate([prefix_attn,suffix_attn],axis=-1)
            positions=jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=1)-1
            (_,output),_=base.PaliGemma.llm([None,suffix],mask=attention,positions=positions,kv_cache=cache,adarms_cond=[None,ad])
            velocity=base.action_out_proj(output[:,-base.action_horizon:])
            baseline=x_t+dt*velocity
            def corrected(v):
                v32=v.astype(jnp.float32)
                guided=guide_velocity(weights,z,x_t,time,v32,target,eta)[0]
                # Preserve the exact original BF16 baseline update. Add the
                # correction in FP32 so small learned gradients are not rounded
                # away when the policy velocity itself is BF16.
                return baseline+jnp.asarray(dt,dtype=jnp.float32)*(guided-v32)
            next_x=jax.lax.cond(enabled & (index%self.guidance_every==0),corrected,lambda _:baseline,velocity)
            return next_x,time+dt,index+1
        x,_,_=jax.lax.while_loop(lambda carry:carry[1]>=-dt/2,step,(noise,jnp.asarray(1.,dtype=jnp.float32),jnp.asarray(0)))
        return x,z


class GuidedPolicy:
    def __init__(self,policy,weights,guidance_every=5):
        self.policy=policy;self.model=CorrectedModel(policy._model,weights,guidance_every)
        self.sample=module_jit(self.model.sample);self.encode_images=module_jit(self.model.encode);self.predict_latent=module_jit(self.model.predict)

    def inputs(self,obs):
        values=jax.tree.map(lambda x:x,obs)
        values=self.policy._input_transform(values)
        return jax.tree.map(lambda x:jnp.asarray(x)[None,...],values)

    def infer(self,obs,*,target=None,enabled=False,eta=1.,noise=None):
        values=self.inputs(obs);observation=model_module.Observation.from_dict(values)
        self.policy._rng,key=jax.random.split(self.policy._rng)
        if noise is None:noise=jax.random.normal(key,(1,50,32))
        elif np.asarray(noise).ndim==2:noise=jnp.asarray(noise)[None,...]
        else:noise=jnp.asarray(noise)
        if target is None:target=np.zeros((512,2048),dtype=np.float32)
        target=jnp.asarray(target,dtype=jnp.float32)
        if target.ndim==2:target=target[None,...]
        actions,z=self.sample(key,observation,noise,target,jnp.asarray(enabled),jnp.asarray(eta,dtype=jnp.float32))
        raw=np.asarray(actions[0]);features=np.asarray(z[0])
        result=self.policy._output_transform(dict(state=np.asarray(values['state'][0]),actions=raw))
        return result['actions'],features,raw

    def encode(self,obs):
        # Requests contain the exact224x224 RGB images from the existing client.
        images=np.stack([obs['cam0'],obs['cam1']])
        if images.shape!=(2,224,224,3) or images.dtype!=np.uint8:raise ValueError('Expected existing pi05 image preprocessing')
        return np.asarray(self.encode_images(jnp.asarray(images,dtype=jnp.float32)/255*2-1)[0])

    def predict(self,z,action):
        return np.asarray(self.predict_latent(jnp.asarray(z,dtype=jnp.float32)[None,...],jnp.asarray(action,dtype=jnp.float32)[None,...])[0])
