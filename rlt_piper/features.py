"""Post-transformer image prefix, matching upstream extract_prefix_embeddings.

Only the two valid cameras are retained; the permanently masked third camera
is excluded. The original VLA parameters and transforms are never modified.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask
from openpi.shared.nnx_utils import module_jit

class Prefix(nnx.Module):
    def __init__(self,base):self.base=base
    def features(self,observation):
        observation=model_module.preprocess_observation(None,observation,train=False)
        x,mask,ar=self.base.embed_prefix(observation)
        n=(x.shape[1]-observation.tokenized_prompt.shape[1])//3
        (out,_),_=self.base.PaliGemma.llm([x,None],mask=make_attn_mask(mask,ar),positions=jnp.cumsum(mask,axis=1)-1)
        return out[:,:2*n].astype(jnp.float32)

class Extractor:
    def __init__(self,policy):self.policy=policy;self.call=module_jit(Prefix(policy._model).features)
    def __call__(self,obs):
        inputs=self.policy._input_transform(jax.tree.map(lambda x:x,obs))
        values=jax.tree.map(lambda x:jnp.asarray(x)[None,...],inputs)
        return self.call(model_module.Observation.from_dict(values))[0]
