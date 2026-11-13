"""Isolated saved-image inference + live learning + hot actor integration check.

No CAN, cameras, keyboard, RPC listeners or formal training assets are opened.
Synthetic learner state is deliberately advanced ONLY inside a temporary folder
so first hot activation can be tested without creating a usable robot actor.
"""
import argparse,json,tempfile,time,threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import jax,jax.numpy as jnp
    from openpi.policies import policy_config
    from openpi_client.image_tools import resize_with_pad
    from rtc_piper.model_adapter import training_config,PROMPT
    from rlt_piper.policy import Policy
    from rlt_piper.recording import Recorder
    from rlt_piper.live import Learner,PriorityGate
    from rlt_piper.serve import Service
    from rlt_piper.contract import assets
    if jax.default_backend()!='gpu':raise RuntimeError('GPU validation required')
    base=policy_config.create_trained_policy(training_config(a.root),a.root/'checkpoints/pi05_piper_2500_bf16',default_prompt=PROMPT,sample_kwargs={'num_steps':10})
    with np.load(a.root/'offline_sample.npz',allow_pickle=False) as d:obs={k:np.array(d[k]) for k in ('cam0','cam1','state','prompt')}
    obs['prompt']=str(obs['prompt'].item())
    for k in ('cam0','cam1'):obs[k]=resize_with_pad(obs[k],224,224)
    with tempfile.TemporaryDirectory(prefix='rlt_hil_gpu_') as folder:
        root=Path(folder);directory=root/'assets';directory.mkdir();(root/'vendor').symlink_to(a.root/'vendor',target_is_directory=True)
        original=a.root/'rlt_assets/pi05_2500'
        for name in ['token.msgpack','token.json','norm_stats.json']:(directory/name).symlink_to(original/name)
        policy=Policy(base,a.root,directory,'online');policy.infer(obs);template=dict(policy.last)
        gate=PriorityGate();learner=Learner(root,directory,policy,gate)
        cfg=dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05)
        rec=Recorder(root,'online',policy.meta['token_sha256'],cfg)
        # Only synthetic replay in a temporary root. It cannot enter the live buffer.
        for i in range(660):
            stamp=10+i*.1;item=dict(template,source='base')
            rec.plan(SimpleNamespace(t=stamp),dict(generation=1,rlt=item),stamp+.14,source='human')
        for stamp in np.arange(9.99,76.1,1/120):rec.command(stamp,obs['state'],'human',1)
        rec.snapshot(live=True);learner.scan()
        if len(learner.rows)<600:raise RuntimeError('Synthetic replay reconstruction failed')
        batch=learner.batch()
        if not np.all(np.asarray(batch['source_chunk'])==2):raise RuntimeError('Human BC labels lost')
        # Verify exact upstream human-action target is active: BC human metric >0.
        t=time.monotonic()
        state,metrics=learner.train_step(learner.state.replace(global_step=jnp.array(1,jnp.int32)),batch,actor=learner.actor,critic=learner.critic,rl_config=learner.cfg,bc_weight=5.,q_weight=0.,delta_weight=0.)
        jax.block_until_ready((state,metrics));step_ms=1000*(time.monotonic()-t)
        if float(metrics['human_mask_ratio'])!=1 or not np.isfinite(float(metrics['bc_human_penalty'])):raise RuntimeError('Human supervision not active')
        # Advance test-only counters to exercise activation boundary; not task training.
        learner.state=state.replace(global_step=jnp.array(1998,jnp.int32),actor_version=jnp.array(999,jnp.int32))
        learner.ready_at=600
        learner.start();service=Service(policy,gate,learner);rtts=[];versions=[]
        try:
            for i in range(40):
                request=dict(method='features' if i%5==0 else 'infer',session='synthetic',request_id=i,generation=1,observed_at=10+i*.033,**obs)
                before=time.monotonic();out=service.infer(request);rtts.append((time.monotonic()-before)*1000);versions.append(out['rlt']['actor_version'])
                if not np.isfinite(out['actions']).all():raise RuntimeError('Nonfinite hot actor output')
                time.sleep(.025)
            if learner.error:raise RuntimeError(learner.error)
            if int(learner.state.global_step)<2020:raise RuntimeError('Insufficient concurrent gradient updates')
            if max(versions)<=0:raise RuntimeError('Hot actor was not activated')
        finally:learner.close()
        from rlt_piper.checkpoint import snapshot
        saved=snapshot(directory,policy.meta['token_sha256'])
        if saved is None or saved[0]['version']<=0:raise RuntimeError('Live learner checkpoint not published')
        report=dict(status='passed',hardware_opened=False,robot_commands_sent=0,synthetic=True,formal_actor_written=False,
                    synthetic_counter_advance=True,transitions=len(learner.rows),human_mask_ratio=float(metrics['human_mask_ratio']),
                    first_gradient_ms=step_ms,concurrent_rpc_p50_ms=float(np.median(rtts)),concurrent_rpc_p95_ms=float(np.percentile(rtts,95)),
                    concurrent_rpc_max_ms=max(rtts),activated_actor_version=max(versions),learner_step=int(learner.state.global_step))
        a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
