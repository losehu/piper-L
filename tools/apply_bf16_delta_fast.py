"""Low-memory CPU delta restore, without extra checksums or equality rechecks.

Only use with the matching BF16 base and its delta package. The normal checked
alternative is apply_bf16_delta.py. Neither path opens robot/camera hardware.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from export_bf16 import restore_bf16,flatten_tree,unflatten_tree,save_params


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--patch',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve();patch=args.patch.resolve()
    if output.exists():raise FileExistsError(output)
    stage=output.with_name(output.name+'.fast-partial');stage.mkdir(parents=True)
    metadata=json.loads((patch/'delta_metadata.json').read_text());started=time.monotonic()
    base_path=args.base/'params' if (args.base/'params').is_dir() else args.base
    base=restore_bf16(base_path);leaves=flatten_tree(base)
    with np.load(patch/'bf16_delta.npz',allow_pickle=False) as arrays:
        for record in metadata['leaves']:
            path=tuple(record['path']);array=leaves[path]
            if not record['unchanged']:
                if not array.flags.writeable:array=array.copy();leaves[path]=array
                xor=arrays[record['key']]
                np.bitwise_xor(array.view(np.uint16),xor,out=array.view(np.uint16))
                del xor
            gc.collect()
    target=unflatten_tree(leaves);del base;gc.collect()
    save_params(stage/'params',target)
    shutil.copytree(patch/'assets',stage/'assets')
    record=dict(source_checkpoint=metadata['source'],base=str(base_path),
                conversion='in-place BF16 XOR on CPU',weight_validation='not requested',
                elapsed_seconds=time.monotonic()-started)
    (stage/'deployment_source.json').write_text(json.dumps(record,indent=2)+'\n')
    os.rename(stage,output)
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
