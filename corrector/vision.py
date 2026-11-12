"""Reuse the selected OpenPI SigLIP image encoder, never another VLA backbone."""
import argparse
import json
from pathlib import Path
import numpy as np


def export_encoder(checkpoint,output):
    import jax
    from flax import traverse_util
    from openpi.models import model
    # Restore to host RAM: exporting the small visual subtree never reserves GPU RAM.
    params=model.restore_params(Path(checkpoint)/'params',restore_type=np.ndarray)
    subtree=params['PaliGemma']['img']
    flat=traverse_util.flatten_dict(subtree);paths=list(flat)
    arrays={'p%d'%i:np.asarray(flat[p],dtype=np.float32) for i,p in enumerate(paths)}
    arrays['paths_json']=np.array(json.dumps(paths))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.partial').open('wb') as f:np.savez(f,**arrays)
    output.with_suffix('.partial').replace(output)
    print(json.dumps(dict(status='exported',parameters=sum(x.size for k,x in arrays.items() if k!='paths_json'),
        checkpoint=str(checkpoint),output=str(output))),flush=True)


def load_encoder(path,device=None):
    import jax
    import jax.numpy as jnp
    from flax import nnx,traverse_util
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.nnx_utils import module_jit
    device=device or jax.devices('gpu')[0]
    with jax.default_device(jax.devices('cpu')[0]):
        template=nnx.eval_shape(Pi0Config(pi05=True,action_horizon=50).create,jax.random.key(0)).PaliGemma.img
    graph,state=nnx.split(template)
    with np.load(path,allow_pickle=False) as data:
        paths=json.loads(str(data['paths_json'].item()))
        values={tuple(p):jax.device_put(jnp.asarray(data['p%d'%i],dtype=jnp.bfloat16),device) for i,p in enumerate(paths)}
    state.replace_by_pure_dict(traverse_util.unflatten_dict(values))
    encoder=nnx.merge(graph,state)
    return module_jit(encoder.__call__,static_argnames=('train',))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();export_encoder(args.checkpoint,args.output)


if __name__=='__main__':main()
