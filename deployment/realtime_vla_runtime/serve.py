"""Local-only Triton policy/Corrector server. Never imports robot hardware."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
from realtime_vla_runtime.policy import Engine,CorrectedEngine


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--step',type=int,choices=[1500,2500],default=2500)
    p.add_argument('--mode',choices=['policy','corrector'],default='policy');p.add_argument('--assets',type=Path)
    args=p.parse_args();root=args.root.resolve()
    from rtc_piper.pi05_options import CHECKPOINTS
    from rtc_piper.pi05_backend import STATES,preflight
    preflight(root,dict(model_step=args.step,checkpoint_index=CHECKPOINTS[args.step][0]),args.mode,args.assets)
    name,port=STATES[args.mode];path=root/'logs'/name
    import fcntl
    with (root/'logs'/('realtime_vla_'+args.mode+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(pid=os.getpid(),argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v],
                   owner_root=str(root),port=port,status='loading',hardware_opened=False)
        path.write_text(json.dumps(state,indent=2)+'\n')
        engine=CorrectedEngine(root,args.assets,args.step) if args.mode=='corrector' else Engine(root,args.step)
        with np.load(root/'offline_sample.npz',allow_pickle=False) as s:
            obs={k:np.array(s[k]) for k in ['cam0','cam1','state']};obs['prompt']=str(s['prompt'].item())
        from openpi_client.image_tools import resize_with_pad
        for key in ('cam0','cam1'):obs[key]=resize_with_pad(obs[key],224,224)
        warm=[];noise=np.random.default_rng(42).normal(size=(50,32)).astype(np.float32)
        baseline,z,_=engine.infer(obs,noise=noise)
        for _ in range(5):
            before=time.monotonic();a,_,_=engine.infer(obs);warm.append(1000*(time.monotonic()-before))
            if a.shape!=(50,7) or not np.isfinite(a).all():raise RuntimeError('Invalid accelerated warmup actions')
        metadata=dict(model='pi05_piper_grasp_%d'%args.step,training_updates=args.step,
                      source_checkpoint_index=CHECKPOINTS[args.step][0],action_horizon=50,physical_action_dim=7,
                      fps=30,warmed_up=True,robot_connected=False,backend='gpu',inference_backend='realtime-vla',
                      upstream_commit=engine.meta['upstream_commit'],num_steps=10,warmups_ms=warm,
                      gpu=engine.torch.cuda.get_device_name(),numerical_validation='saved observations, no real grasp validation')
        if args.mode=='corrector':
            from corrector.contract import VERSION,identity
            from corrector.serve import Service
            meta=json.loads((args.assets/'metadata.json').read_text());norms=json.loads((args.assets/'norm_stats.json').read_text());norms=norms.get('norm_stats',norms)
            if meta['contract']!=identity(norms,meta['contract']['k_step']):raise RuntimeError('Corrector contract mismatch')
            base_norms=json.loads((root/CHECKPOINTS[args.step][1]/'assets/local/piper_grasp_train/norm_stats.json').read_text());base_norms=base_norms.get('norm_stats',base_norms)
            if identity(base_norms,meta['contract']['k_step'])!=meta['contract']:raise RuntimeError('Corrector/base normalization mismatch')
            target=engine.predict(z,np.zeros(7,dtype=np.float32))
            zero=engine.infer(obs,noise=noise,target=target,enabled=True,eta=0)[0]
            guided=engine.infer(obs,noise=noise,target=target,enabled=True)[0]
            delta=abs(guided-baseline)
            if not np.array_equal(zero,baseline):raise RuntimeError('Zero OGG coefficient changed baseline')
            if not np.isfinite(guided).all() or not np.any(delta):raise RuntimeError('OGG branch had no finite action effect')
            engine.encode(obs)
            metadata.update(protocol=VERSION,model='pi05_piper_grasp_2500_corrector',
                            contract=meta['contract'],monitor_config=meta['monitor_config'],
                            corrector_epoch=meta['epoch'],corrector_parameters=meta['parameters'],
                            zero_eta_exact=True,guide_steps_per_recovery=2,
                            guided_max_joint_delta_rad=float(delta[:,:6].max()),guided_max_gripper_delta_m=float(delta[:,6].max()))
            service=Service(engine,metadata,norms)
        else:
            class Policy:
                failed=False
                def infer(self,request):
                    if self.failed:raise RuntimeError('Policy error latched; restart service')
                    try:
                        values=np.asarray(request['state'])
                        if values.shape!=(7,) or not np.isfinite(values).all():raise ValueError('Invalid finite7D state')
                        obs={k:request[k] for k in ['cam0','cam1','state','prompt']}
                        action,_,_=engine.infer(obs)
                        if action.shape!=(50,7) or not np.isfinite(action).all():raise RuntimeError('Invalid accelerated actions')
                        return dict(actions=action,request_id=request.get('request_id',0),observed_at=request.get('observed_at',0))
                    except BaseException:self.failed=True;raise
            service=Policy()
        state.update(status='ready',metadata=metadata);path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state),flush=True)
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        server=WebsocketPolicyServer(service,host='127.0.0.1',port=port,metadata=metadata)
        import asyncio
        async def graceful_server(server):
            from websockets.asyncio.server import serve
            from openpi.serving.websocket_policy_server import _health_check
            loop=asyncio.get_running_loop();stopped=asyncio.Event()
            for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stopped.set)
            async with serve(server._handler,server._host,server._port,compression=None,max_size=None,process_request=_health_check):
                await stopped.wait()
        asyncio.run(graceful_server(server))

if __name__=='__main__':main()
