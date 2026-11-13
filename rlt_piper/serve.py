"""Frozen-pi05 RLT server. No Corrector or hardware is loaded."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from rlt_piper.contract import assets,SCHEMA
from rlt_piper.policy import Policy

class Service:
    def __init__(self,policy):self.policy=policy
    def infer(self,request):
        method=request.get('method')
        response=dict(protocol=SCHEMA,request_id=request['request_id'],session=request['session'],ok=True,generation=0)
        if method=='status':return response
        if method not in ('infer','features'):raise ValueError('Unknown RLT request')
        if not np.isfinite(float(request['observed_at'])):raise ValueError('Invalid observation time')
        obs={k:request[k] for k in ['cam0','cam1','state','prompt']}
        actions=self.policy.infer(obs)
        if actions.shape!=(50,7) or not np.isfinite(actions).all():raise RuntimeError('Invalid RLT output')
        return dict(response,actions=actions,rlt=self.policy.last,observed_at=request['observed_at'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--stage',choices=['collect','eval','online'],required=True);args=p.parse_args()
    root=args.root.resolve();directory=args.assets.resolve();m=assets(directory,args.stage!='collect')
    import jax
    from openpi.policies import policy_config
    from rtc_piper.model_adapter import training_config,PROMPT
    if jax.default_backend()!='gpu':raise RuntimeError('RLT requires the configured GPU')
    checkpoint=root/'checkpoints/pi05_piper_2500_bf16'
    norms=json.loads((checkpoint/'assets/local/piper_grasp_train/norm_stats.json').read_text())
    learned_norms=json.loads((directory/'norm_stats.json').read_text())
    if norms.get('norm_stats',norms)!=learned_norms.get('norm_stats',learned_norms):raise RuntimeError('RLT/base normalization mismatch')
    base=policy_config.create_trained_policy(training_config(root),checkpoint,default_prompt=PROMPT,sample_kwargs={'num_steps':10})
    policy=Policy(base,root,directory,args.stage)
    with np.load(root/'offline_sample.npz',allow_pickle=False) as s:obs={k:np.array(s[k]) for k in ('cam0','cam1','state','prompt')}
    obs['prompt']=str(obs['prompt'].item())
    from openpi_client.image_tools import resize_with_pad
    for key in ('cam0','cam1'):obs[key]=resize_with_pad(obs[key],224,224)
    timings=[]
    for _ in range(3):
        before=time.monotonic();out=policy.infer(obs);timings.append(time.monotonic()-before)
        if out.shape!=(50,7) or not np.isfinite(out).all():raise RuntimeError('Invalid warmed RLT output')
    metadata=dict(protocol=SCHEMA,model='pi05_piper_rlt',base_step=2500,base_frozen=True,rlt_stage=args.stage,
                  rlt_actor_version=policy.version,rlt_chunk_len=18,rlt_token_sha256=m['token_sha256'],
                  action_horizon=50,physical_action_dim=7,fps=30,warmed_up=True,corrector_enabled=False,
                  rlt_warmups_s=timings,robot_connected=False)
    state=dict(pid=os.getpid(),argv=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x],
               owner_root=str(root),status='ready',port=8003,metadata=metadata,hardware_opened=False)
    (root/'logs/rlt_service.json').write_text(json.dumps(state,indent=2)+'\n')
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(Service(policy),host='127.0.0.1',port=8003,metadata=metadata).serve_forever()

if __name__=='__main__':main()
