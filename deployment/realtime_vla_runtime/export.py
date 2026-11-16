"""CPU-only, derived BF16 asset export. Original OpenPI checkpoint is read-only."""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import time

UPSTREAM='eb46c8d8ccf5216fd4bb71a48429f2da9dcf36a0'


def allocation_shapes(path):
    """Read the upstream static shape table without executing its CUDA CLI."""
    tree=ast.parse(Path(path).read_text())
    tables=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='weights' for t in n.targets)
            and isinstance(n.value,ast.Dict)]
    table=tables[-1];out={}
    for key,value in zip(table.keys,table.values):
        if not (isinstance(key,ast.Constant) and isinstance(value,ast.Call)
                and isinstance(value.func,ast.Attribute) and value.func.attr=='zeros'):
            raise ValueError('Unexpected upstream weight table')
        def dimension(x):
            if isinstance(x,ast.Constant) and isinstance(x.value,int):return x.value
            if isinstance(x,ast.Name) and x.id=='prompt_len':return 0
            if isinstance(x,ast.BinOp) and isinstance(x.op,ast.Mult):return dimension(x.left)*dimension(x.right)
            raise ValueError('Unexpected weight shape')
        dims=[dimension(x) for x in value.args]
        out[key.value]=tuple(dims)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,choices=[1500,2500],default=2500);args=p.parse_args()
    os.environ['JAX_PLATFORMS']='cpu'
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from vendor.realtime_vla import convert_from_jax_pi05 as converter
    from rtc_piper.pi05_options import CHECKPOINTS
    root=args.root.resolve();index,relative=CHECKPOINTS[args.step]
    source=root/relative;dest=root/'realtime_vla_assets'/('pi05_%d'%args.step)
    if not (source/'params').is_dir():raise RuntimeError('Original checkpoint missing: '+str(source))
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'model.safetensors').exists():raise RuntimeError('Derived asset exists; export will not overwrite it')
    torch.set_num_threads(4);started=time.monotonic()
    dump=converter.load_jax_weights(str(source))
    def host(node):
        if isinstance(node,dict):return {k:host(v) for k,v in node.items()}
        return np.asarray(node)
    dump=host(dump)
    shapes=allocation_shapes(Path(converter.__file__))
    weights={k:torch.empty(shape,dtype=torch.bfloat16,device='cpu') for k,shape in shapes.items()}
    converter.convert_weights_pi05(weights,dump)
    weights['embedding_weight'].copy_(torch.from_numpy(
        np.asarray(dump['PaliGemma']['llm']['embedder']['input_embedding']['value'],dtype=np.float32)).to(torch.bfloat16))
    time_value=torch.tensor(1.,dtype=torch.float32);dt=-1./10
    period=4e-3*(4./4e-3)**torch.linspace(0.,1.,512)
    rows=[]
    for _ in range(10):
        angle=time_value/period*2*np.pi
        rows.append(torch.cat((torch.sin(angle),torch.cos(angle))).to(torch.bfloat16));time_value=time_value+dt
    weights['decoder_time_embeds'].copy_(torch.stack(rows))
    del dump;gc.collect()
    temp=dest/'model.safetensors.partial';save_file(weights,str(temp));temp.replace(dest/'model.safetensors')
    norms=(source/'assets/local/piper_grasp_train/norm_stats.json').read_bytes()
    meta=dict(format='piper-realtime-vla-v1',upstream_commit=UPSTREAM,training_updates=args.step,
              checkpoint_index=index,source_checkpoint=relative,num_steps=10,action_horizon=50,action_dim=32,
              valid_camera_order=['base_0_rgb','left_wrist_0_rgb'],masked_camera='right_wrist_0_rgb',
              norm_sha256=hashlib.sha256(norms).hexdigest(),export_seconds=time.monotonic()-started,
              original_checkpoint_modified=False,hardware_opened=False)
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
