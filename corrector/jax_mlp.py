"""JAX equivalent of the pinned residual MLP; no backbone parameter gradients."""
import json
from pathlib import Path
import numpy as np


def load(path):
    import jax.numpy as jnp
    with np.load(path,allow_pickle=False) as data:return {k:jnp.asarray(data[k]) for k in data.files}


def forward(weights,z,action):
    import jax
    import jax.numpy as jnp
    def linear(x,name):return jnp.matmul(x,weights[name+'.weight'].T,precision=jax.lax.Precision.HIGHEST)+weights[name+'.bias']
    def norm(x,name):
        mean=jnp.mean(x,axis=-1,keepdims=True);variance=jnp.mean((x-mean)**2,axis=-1,keepdims=True)
        return (x-mean)*jax.lax.rsqrt(variance+1e-5)*weights[name+'.weight']+weights[name+'.bias']
    z=jnp.asarray(z,dtype=jnp.float32);action=jnp.asarray(action,dtype=jnp.float32)
    embedding=linear(action,'action_proj')
    embedding=jnp.broadcast_to(embedding[...,None,:],(*z.shape[:-1],embedding.shape[-1]))
    x=linear(jnp.concatenate([z,embedding],axis=-1),'in_proj')
    count=sum(k.endswith('.fc1.weight') for k in weights)
    for i in range(count):
        name='blocks.%d'%i;y=linear(norm(x,name+'.norm'),name+'.fc1');y=jax.nn.silu(y)
        x=x+linear(y,name+'.fc2')
        transition='transitions.%d'%i
        if transition+'.0.weight' in weights:x=jax.nn.silu(linear(norm(x,transition+'.0'),transition+'.1'))
    return linear(norm(x,'out_norm'),'out_proj')


def guide_velocity(weights,z,x_t,time,velocity,target,eta):
    import jax
    import jax.numpy as jnp
    # Match the original detached velocity leaf. This differentiates the MLP
    # and clean-action estimate, never the frozen vision/VLA parameters.
    detached=jax.lax.stop_gradient(velocity)
    def objective(v):
        clean=x_t-time*v;candidate=clean[:,0,:7]
        predicted=forward(weights,z,candidate).reshape((z.shape[0],-1))
        desired=target.reshape((z.shape[0],-1))
        cosine=jnp.sum(predicted*desired,axis=1)/(jnp.maximum(jnp.linalg.norm(predicted,axis=1),1e-8)*jnp.maximum(jnp.linalg.norm(desired,axis=1),1e-8))
        return jnp.mean(1-cosine)
    loss,gradient=jax.value_and_grad(objective)(detached)
    return detached-eta*gradient,loss,jnp.linalg.norm(gradient)
