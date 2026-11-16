"""Saved-observation JAX/Triton validation and timings; never hardware."""
import argparse,json,os,time
from pathlib import Path
import numpy as np


def observations(root):
    paths=[root/'offline_sample.npz']
    candidates=list((root/'logs').glob('*observation*.npz'))+list((root/'logs').glob('policy_rejected_*.npz'))
    paths+=sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True)[:5]
    from openpi_client.image_tools import resize_with_pad
    out=[]
    for path in paths:
        try:
            with np.load(path,allow_pickle=False) as s:
                obs={key:np.array(s[key]) for key in ('cam0','cam1','state')}
                obs['prompt']=str(s['prompt'].item()) if 'prompt' in s else 'Pick up all the objects on the table and place them in the box.'
            for key in ('cam0','cam1'):obs[key]=resize_with_pad(obs[key],224,224)
            out.append((path.name,obs))
        except (KeyError,ValueError):continue
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['reference','triton'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--step',type=int,default=2500)
    args=p.parse_args();root=args.root.resolve();logs=root/'logs';logs.mkdir(exist_ok=True)
    path=logs/('realtime_vla_reference_%d.npz'%args.step)
    if args.phase=='reference':
        import jax,jax.numpy as jnp
        from flax import nnx
        from openpi.policies import policy_config
        from openpi.models.model import Observation,preprocess_observation
        from openpi.shared.nnx_utils import module_jit
        from rtc_piper.model_adapter import training_config,PROMPT
        from rtc_piper.pi05_options import CHECKPOINTS
        base=policy_config.create_trained_policy(training_config(root),root/CHECKPOINTS[args.step][1],
                                                default_prompt=PROMPT,sample_kwargs={'num_steps':10})
        class Features(nnx.Module):
            def __init__(self,model):self.model=model
            def encode(self,obs):return self.model.embed_prefix(preprocess_observation(None,obs,train=False))[0][:,:512]
        encode=module_jit(Features(base._model).encode)
        records={};timings=[];n=0
        for name,obs in observations(root):
            values=base._input_transform(dict(obs))
            z=np.asarray(encode(Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None,...],values))),dtype=np.float32)[0]
            for seed in [42,17,103]:
                noise=np.random.default_rng(seed).normal(size=(50,32)).astype(np.float32)
                before=time.monotonic();a=base.infer(dict(obs),noise=noise)['actions'];elapsed=time.monotonic()-before
                if n:timings.append(elapsed)
                for key,value in obs.items():records['%d_%s'%(n,key)]=np.asarray(value)
                records['%d_noise'%n]=noise;records['%d_actions'%n]=a;records['%d_z'%n]=z
                records['%d_source'%n]=np.array(name);n+=1
        records['count']=np.array(n);records['timings_s']=np.array(timings)
        np.savez_compressed(path,**records);print(json.dumps(dict(status='reference_saved',cases=n,timings_ms=[1000*x for x in timings])),flush=True)
    else:
        from realtime_vla_runtime.policy import Engine
        engine=Engine(root,args.step);errors=[];latent_errors=[];timings=[]
        with np.load(path,allow_pickle=False) as data:
            for i in range(int(data['count'])):
                obs={key:np.array(data['%d_%s'%(i,key)]) for key in ['cam0','cam1','state']}
                obs['prompt']=str(data['%d_prompt'%i].item());noise=data['%d_noise'%i]
                for _ in range(2):
                    before=time.monotonic();a,z,raw=engine.infer(obs,noise=noise);timings.append(time.monotonic()-before)
                errors.append(a-data['%d_actions'%i])
                reference_z=data['%d_z'%i]
                if reference_z.dtype.kind=='V':
                    import ml_dtypes
                    reference_z=reference_z.view(ml_dtypes.bfloat16).astype(np.float32)
                latent_errors.append(float(np.linalg.norm(z-reference_z)/max(np.linalg.norm(reference_z),1e-8)))
            baseline=data['timings_s']
        error=np.array(errors);joint=np.degrees(abs(error[:,:,:6]));grip=abs(error[:,:,6])*1000
        passed=(float(np.percentile(joint,95))<=1 and float(np.max(joint))<=3
                and float(np.percentile(grip,95))<=2 and float(np.max(grip))<=5 and max(latent_errors)<=.03)
        result=dict(status='passed' if passed else 'failed',hardware_opened=False,robot_commands_sent=0,
                    cases=len(errors),joint_error_p95_deg=float(np.percentile(joint,95)),joint_error_max_deg=float(np.max(joint)),
                    gripper_error_p95_mm=float(np.percentile(grip,95)),gripper_error_max_mm=float(np.max(grip)),
                    latent_relative_l2=latent_errors,triton_p50_ms=1000*float(np.median(timings)),
                    triton_p95_ms=1000*float(np.percentile(timings,95)),jax_p50_ms=1000*float(np.median(baseline)),
                    acceptance='numerical saved-observation check only, not real grasp validation')
        (logs/('realtime_vla_validation_%d.json'%args.step)).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not passed:raise RuntimeError('Triton/JAX numerical validation failed')

if __name__=='__main__':main()
