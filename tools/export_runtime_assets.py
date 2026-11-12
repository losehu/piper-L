"""Generate private startup poses and a warmup observation from private HDF5s."""
import argparse
import io
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=root/'training/dataset_manifest.json')
    p.add_argument('--output',type=Path,default=root/'deployment')
    args=p.parse_args()
    spec=json.loads(args.manifest.read_text());starts=[];sample=None
    for row in spec['records']:
        with h5py.File(args.selected_dir/row['source'],'r') as f:
            frame=int(row['start'])
            state=np.asarray(f['observations/qpos'][frame],dtype=np.float32)
            if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Invalid start state')
            starts.append(state.tolist())
            if sample is None and row['partition']=='train':
                sample=dict(state=state,prompt=np.array(spec['task']))
                for cam in ['cam0','cam1']:
                    with Image.open(io.BytesIO(bytes(f['observations/images/'+cam][frame]))) as image:
                        sample[cam]=np.array(image.convert('RGB'))
    if not starts or sample is None:raise ValueError('No training observations found')
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[args.output/'training_start_states.json',args.output/'offline_sample.npz']
    if any(path.exists() for path in paths):raise FileExistsError('Refusing to overwrite existing private runtime assets')
    paths[0].write_text(json.dumps(starts,indent=2)+'\n')
    np.savez_compressed(paths[1],**sample)
    print('Wrote private runtime assets; these files are excluded from Git.')


if __name__=='__main__':main()
