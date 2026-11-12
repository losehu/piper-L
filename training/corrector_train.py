"""Eight-rank DDP training on episode-separated frozen OpenPI features."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from corrector.vendor.vla_corrector.MLP import SiglipResidualMLP
from corrector.vendor.vla_corrector.config import SiglipMLPConfig


class Pairs(Dataset):
    def __init__(self,root,partition,k):
        self.rows=[];self.arrays=[];self.contract=None
        for path in sorted(Path(root).glob(partition+'_*/complete.json')):
            row=json.loads(path.read_text());contract=row['contract']
            if self.contract is not None and contract!=self.contract:raise ValueError('Mixed feature contracts')
            self.contract=contract
            z=np.load(path.parent/'z.npy',mmap_mode='r');a=np.load(path.parent/'actions.npy',mmap_mode='r')
            if z.shape[0]!=a.shape[0] or contract['k_step']!=k:raise ValueError('Feature time/action mismatch')
            e=len(self.arrays);self.arrays.append((z,a))
            self.rows.extend((e,t) for t in range(len(z)-k))
        if not self.rows:raise ValueError('No complete features for '+partition)
        self.k=k;self.token_dim=self.arrays[0][0].shape[-1]

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        e,t=self.rows[index];z,a=self.arrays[e]
        current=np.array(z[t],dtype=np.float32)
        delta=np.array(z[t+self.k],dtype=np.float32)-current
        return torch.from_numpy(current),torch.from_numpy(np.array(a[t],copy=True)),torch.from_numpy(delta)


def losses(pred,target):
    p=pred.float().flatten(1);y=target.float().flatten(1)
    cosine=1-torch.nn.functional.cosine_similarity(p,y,dim=1,eps=1e-8)
    return cosine,(p-y).square().mean(1),y.norm(dim=1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--k-step',type=int,default=5);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--microbatch',type=int,default=8);p.add_argument('--global-batch',type=int,default=512)
    p.add_argument('--memory-fraction',type=float,default=.10);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('Requested run requires all8 H100 ranks')
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[local*8:local*8+8])
    torch.cuda.set_device(local);torch.cuda.set_per_process_memory_fraction(args.memory_fraction,local)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    dist.init_process_group('nccl');device=torch.device('cuda',local)
    if args.global_batch%(world*args.microbatch):raise ValueError('Global batch must divide across ranks and microbatches')
    accumulation=args.global_batch//(world*args.microbatch)
    train=Pairs(args.features,'train',args.k_step);val=Pairs(args.features,'val',args.k_step)
    if train.contract!=val.contract:raise ValueError('Train/val contracts differ')
    model=SiglipResidualMLP(SiglipMLPConfig(token_dim=train.token_dim,action_dim=7)).to(device)
    parameters=sum(p.numel() for p in model.parameters())
    model=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=3e-6)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=42,drop_last=False)
    loader=DataLoader(train,batch_size=args.microbatch,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    # Uneven validation partitions avoid duplicated held-out examples. No DDP
    # forward collectives are used during this phase.
    valid=DataLoader(torch.utils.data.Subset(val,list(range(rank,len(val),world))),batch_size=args.microbatch,
        num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    args.output.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');bad=0;begun=time.monotonic()
    if rank==0:
        (args.output/'run.json').write_text(json.dumps(dict(args=vars(args)|{'features':str(args.features),'output':str(args.output)},
            world_size=world,parameters=parameters,contract=train.contract,train_pairs=len(train),validation_pairs=len(val),
            train_episodes=len(train.arrays),validation_episodes=len(val.arrays),accumulation=accumulation),indent=2)+'\n')
        print(json.dumps(dict(status='training',world_size=world,parameters=parameters,global_batch=args.global_batch,
            microbatch_per_gpu=args.microbatch,accumulation=accumulation,train_pairs=len(train),val_pairs=len(val))),flush=True)
    try:
        for epoch in range(1,args.epochs+1):
            sampler.set_epoch(epoch);model.train();optimizer.zero_grad(set_to_none=True)
            totals=torch.zeros(4,device=device,dtype=torch.float64);group_size=accumulation
            for step,(z,a,target) in enumerate(loader):
                if step%accumulation==0:group_size=min(accumulation,len(loader)-step)
                synchronize=(step%accumulation==group_size-1 or step+1==len(loader))
                context=nullcontext() if synchronize else model.no_sync()
                z=z.to(device,non_blocking=True);a=a.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
                with context,torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(z,a);cos,mse,norm=losses(pred,target);loss=cos.mean()/group_size
                    loss.backward()
                totals+=torch.stack([cos.detach().sum(),mse.detach().sum(),norm.detach().sum(),torch.as_tensor(len(z),device=device)]).double()
                if synchronize:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
                if rank==0 and (step+1)%100==0:
                    print(json.dumps(dict(epoch=epoch,microstep=step+1,total_microsteps=len(loader),elapsed_s=time.monotonic()-begun)),flush=True)
            dist.all_reduce(totals);model.eval();validation=torch.zeros(4,device=device,dtype=torch.float64);scores=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for z,a,target in valid:
                    z=z.to(device,non_blocking=True);a=a.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
                    cos,mse,norm=losses(model.module(z,a),target)
                    scores.append(cos.float().cpu().numpy())
                    validation+=torch.stack([cos.sum(),mse.sum(),norm.sum(),torch.as_tensor(len(z),device=device)]).double()
            dist.all_reduce(validation);scheduler.step();score=float(validation[0]/validation[3])
            improved=score<best-1e-5
            if improved:
                best=score;bad=0
                np.save(args.output/('best_scores_rank%d.npy'%rank),np.concatenate(scores),allow_pickle=False)
            else:bad+=1
            row=dict(epoch=epoch,train_cosine_loss=float(totals[0]/totals[3]),val_cosine_loss=score,
                val_mse=float(validation[1]/validation[3]),val_target_norm=float(validation[2]/validation[3]),
                best_val=best,elapsed_s=time.monotonic()-begun,lr=scheduler.get_last_lr()[0],
                peak_allocated_GB=torch.cuda.max_memory_allocated()/1e9)
            history.append(row)
            if rank==0:
                print(json.dumps(row),flush=True)
                payload=dict(state_dict={k:v.detach().cpu() for k,v in model.module.state_dict().items()},
                    contract=train.contract,epoch=epoch,metrics=row,parameters=parameters,
                    model_config=dict(token_dim=train.token_dim,action_dim=7,action_embed_dim=256,widths=[2048]*4))
                torch.save(payload,args.output/'last.pt')
                if improved:torch.save(payload,args.output/'best.pt')
                (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            stop=torch.tensor(int(bad>=args.patience),device=device);dist.broadcast(stop,0)
            if stop.item():break
        if rank==0:
            (args.output/'complete.json').write_text(json.dumps(dict(status='trained',best_val=best,epochs=len(history),
                world_size=world,elapsed_s=time.monotonic()-begun,physical_robot_tested=False),indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
