"""realtime-vla Triton core with the trained OpenPI transforms and tokenizer."""
from pathlib import Path
import json
import numpy as np


def compact_inputs(values):
    names=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
    masks=[bool(np.asarray(values['image_mask'][name])) for name in names]
    if masks!=[True,True,False]:raise ValueError('Expected two valid Piper views and masked third camera')
    images=np.stack([np.asarray(values['image'][name]) for name in names[:2]])
    if images.shape!=(2,224,224,3):raise ValueError('Expected transformed 224x224 RGB views')
    if images.dtype==np.uint8:images=images.astype(np.float32)/255*2-1
    elif not np.isfinite(images).all():raise ValueError('Nonfinite transformed images')
    ids=np.asarray(values['tokenized_prompt']);mask=np.asarray(values['tokenized_prompt_mask'],dtype=bool)
    count=int(mask.sum())
    if ids.shape!=(200,) or not np.array_equal(mask,np.arange(200)<count) or not count:
        raise ValueError('Expected contiguous valid OpenPI prompt tokens')
    return images,ids[:count]


def transforms(root,step):
    from rtc_piper.model_adapter import training_config,PROMPT
    from openpi import transforms as t
    from openpi.training.checkpoints import load_norm_stats
    from rtc_piper.pi05_options import CHECKPOINTS
    cfg=training_config(root);data=cfg.data.create(cfg.assets_dirs,cfg.model)
    checkpoint=Path(root)/CHECKPOINTS[step][1]
    norms=load_norm_stats(checkpoint/'assets',data.asset_id)
    inputs=t.compose([t.InjectDefaultPrompt(PROMPT),*data.data_transforms.inputs,
                      t.Normalize(norms,use_quantiles=data.use_quantile_norm),*data.model_transforms.inputs])
    outputs=t.compose([*data.model_transforms.outputs,t.Unnormalize(norms,use_quantiles=data.use_quantile_norm),
                       *data.data_transforms.outputs])
    return inputs,outputs


class Engine:
    def __init__(self,root,step=2500):
        import torch
        from safetensors.torch import load_file
        from vendor.realtime_vla import pi05_infer as kernels
        from rtc_piper.pi05_options import CHECKPOINTS
        from realtime_vla_runtime.export import UPSTREAM
        import hashlib
        if not torch.cuda.is_available():raise RuntimeError('realtime-vla requires CUDA')
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        self.torch,self.kernels=torch,kernels;self.root=Path(root);self.step=step
        directory=self.root/'realtime_vla_assets'/('pi05_%d'%step)
        self.meta=json.loads((directory/'metadata.json').read_text())
        norm=(self.root/CHECKPOINTS[step][1]/'assets/local/piper_grasp_train/norm_stats.json').read_bytes()
        if (self.meta['upstream_commit']!=UPSTREAM or self.meta['training_updates']!=step
            or self.meta['checkpoint_index']!=CHECKPOINTS[step][0]
            or self.meta['norm_sha256']!=hashlib.sha256(norm).hexdigest()):
            raise ValueError('realtime-vla asset/checkpoint/normalization identity mismatch')
        checkpoint=load_file(str(directory/'model.safetensors'),device='cpu')
        # Use original OpenPI token IDs directly; no Hugging Face tokenizer or
        # changed state binning. Fixed allocation includes masked prompt padding.
        embedding=checkpoint.pop('embedding_weight').to('cuda')
        self.out_w=checkpoint['decoder_action_out_proj_w'].to('cuda')
        self.out_b=checkpoint['decoder_action_out_proj_b'].to('cuda')
        checkpoint['language_embeds']=torch.zeros((200,2048),dtype=torch.bfloat16)
        engine=self
        class Core(kernels.Pi05Inference):
            def record_infer_graph(core):
                # CPU/GPU adapter captures only after FP32 diffusion and latent
                # snapshots are installed. Original kernels/weights stay frozen.
                pass
        self.core=Core(checkpoint,2,50,discrete_state_input=False,optimized=False)
        del checkpoint
        self.embedding=embedding
        self.inputs,self.outputs=transforms(self.root,step)
        b=self.core.buffers
        b['diffusion_noise']=torch.empty((50,32),dtype=torch.float32,device='cuda')
        b['vision_prefix_snapshot']=torch.empty((512,2048),dtype=torch.bfloat16,device='cuda')
        # Keep the velocity projection unscaled, then reproduce the original
        # BF16 delta product plus FP32 diffusion accumulator (also supports OGG).
        self.velocity=torch.empty((50,32),dtype=torch.float32,device='cuda')
        b['action_update']=self.action_update
        b['observation_images_normalized'].zero_();b['diffusion_noise'].zero_()
        b['encoder_x'].zero_();b['valid_encoder_len'].fill_(513)
        self.graph=self.capture(lambda:kernels.pi05_model(self.core.weights,b,2,self.core.encoder_seq_len))
        self.encoder_graph=None;self.step_graphs=None

    def action_update(self,weights,buffers,step):
        t=self.torch;k=self.kernels
        k.adarms_norm_style_proj_final(buffers['decoder_x'],buffers['decoder_time_emb'][step],
            weights['decoder_final_norm_mod_w'],weights['decoder_final_norm_mod_b'],buffers['x_normed_buf'],
            buffers['gate_buf'],buffers['decoder_style_final'][step])
        velocity=t.matmul(buffers['x_normed_buf'],self.out_w)+self.out_b
        self.velocity.copy_(velocity)
        buffers['diffusion_noise'].add_((velocity*(-.1)).to(t.bfloat16).float())

    def capture(self,run):
        t=self.torch;main=t.cuda.current_stream();stream=t.cuda.Stream();stream.wait_stream(main)
        with t.cuda.stream(stream),t.no_grad():
            for _ in range(3):run()
        stream.synchronize();graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph,stream=stream),t.no_grad():run()
        main.wait_stream(stream);return graph

    def prepare(self,obs,noise=None):
        t=self.torch;values=self.inputs({k:v for k,v in obs.items()});images,ids=compact_inputs(values)
        if noise is None:noise=np.random.default_rng().normal(size=(50,32)).astype(np.float32)
        noise=np.asarray(noise,dtype=np.float32)
        if noise.shape!=(50,32) or not np.isfinite(noise).all():raise ValueError('Invalid 50x32 diffusion noise')
        b=self.core.buffers;count=len(ids);start=512
        b['observation_images_normalized'].copy_(t.from_numpy(np.ascontiguousarray(images)).to('cuda'))
        b['diffusion_noise'].copy_(t.from_numpy(noise).to('cuda'))
        embeds=t.nn.functional.embedding(t.from_numpy(ids.astype(np.int64)).to('cuda'),self.embedding)*(2048**.5)
        b['encoder_x'][start:start+count].copy_(embeds)
        b['valid_encoder_len'].fill_(start+count)
        b['decoder_rope_weights'].copy_(self.core.get_decoder_rope_weights(count))
        return values

    def result(self,values):
        raw=self.core.buffers['diffusion_noise'].detach().cpu().numpy().copy()
        z=self.core.buffers['vision_prefix_snapshot'].float().cpu().numpy().copy()
        result=self.outputs(dict(state=values['state'],actions=raw))
        return np.asarray(result['actions'],dtype=np.float32),z,raw

    def infer(self,obs,*,noise=None,**kwargs):
        with self.torch.no_grad():
            values=self.prepare(obs,noise);self.graph.replay();return self.result(values)

    def capture_recovery(self):
        if self.step_graphs is not None:return
        k=self.kernels;b=self.core.buffers
        def encoder():
            k.vision_encoder(self.core.weights,b,2)
            k.transformer_encoder(self.core.weights,b,self.core.encoder_seq_len)
        self.encoder_graph=self.capture(encoder)
        self.step_graphs=[self.capture(lambda step=step:k.transformer_decoder(
            self.core.weights,b,self.core.encoder_seq_len,num_steps=1,start_step=step)) for step in range(10)]

    def encode(self,obs):
        t=self.torch;k=self.kernels;b=self.core.buffers
        images,_=compact_inputs(self.inputs({key:value for key,value in obs.items()}))
        with t.no_grad():
            b['observation_images_normalized'].copy_(t.from_numpy(np.ascontiguousarray(images)).to('cuda'))
            k.vision_encoder(self.core.weights,b,2)
            k.layer_norm_matmul_n256_1152_2048_bias(b['vision_x'],self.core.weights['vision_final_norm_w'],
                self.core.weights['vision_final_norm_b'],self.core.weights['encoder_multi_modal_projector_w'],
                self.core.weights['encoder_multi_modal_projector_b'],b['encoder_x'],b['vision_x_norm'])
            return b['encoder_x'][:512].float().cpu().numpy().copy()


class CorrectedEngine(Engine):
    """Same detached-velocity OGG objective, through only the small frozen MLP."""
    def __init__(self,root,assets,step=2500):
        super().__init__(root,step)
        from corrector.vendor.vla_corrector.MLP import SiglipResidualMLP
        from corrector.vendor.vla_corrector.config import SiglipMLPConfig
        meta=json.loads((Path(assets)/'metadata.json').read_text())
        model=SiglipResidualMLP(SiglipMLPConfig(token_dim=meta['model_config']['token_dim'],action_dim=7))
        with np.load(Path(assets)/'mlp.npz',allow_pickle=False) as data:
            model.load_state_dict({key:self.torch.from_numpy(np.array(data[key],copy=True)) for key in data.files})
        self.mlp=model.to('cuda').eval().requires_grad_(False)
        self.capture_recovery()

    def predict(self,z,action):
        t=self.torch
        with t.no_grad():return self.mlp(t.as_tensor(z,device='cuda',dtype=t.float32),
                                         t.as_tensor(action,device='cuda',dtype=t.float32)).cpu().numpy()

    def infer(self,obs,*,noise=None,target=None,enabled=False,eta=1.):
        if not enabled or eta==0:return super().infer(obs,noise=noise)
        t=self.torch;b=self.core.buffers
        if not np.isfinite(eta) or eta<0:raise ValueError('Invalid OGG coefficient')
        with t.no_grad():values=self.prepare(obs,noise);self.encoder_graph.replay()
        z=b['vision_prefix_snapshot'].detach().float()
        desired=t.as_tensor(target,device='cuda',dtype=t.float32).reshape(1,-1)
        for step,graph in enumerate(self.step_graphs):
            with t.no_grad():previous=b['diffusion_noise'][0,:7].clone();graph.replay()
            if step%5:continue
            time_value=1.-step/10
            clean=(previous-time_value*self.velocity[0,:7]).detach().requires_grad_(True)
            predicted=self.mlp(z,clean).reshape(1,-1)
            cosine=t.nn.functional.cosine_similarity(predicted,desired,dim=1,eps=1e-8)
            gradient=t.autograd.grad((1-cosine).mean(),clean)[0]
            with t.no_grad():b['diffusion_noise'][0,:7].add_(-.1*eta*time_value*gradient)
        with t.no_grad():return self.result(values)
