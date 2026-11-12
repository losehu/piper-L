"""Portable per-row INT8 video weights with BF16 compute and staged auxiliaries.

No custom CUDA extension, no FP8 kernels and no Tensor subclasses. Quantization
is symmetric [-127,127] per output row; it is not bitwise TorchAO's API. Only
weight representation changes. Attention, conditioning and action sampling are
inherited unchanged from the pinned official model.
"""
from __future__ import annotations
import gc
import json
import logging
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from vpp2.libero.action_model import LiberoActionModel
from vpp2.models.wan21_14b.action_dit import ActionDiT
from vpp2.models.wan21_14b.wan_video_dit import WanVideoDiT, precompute_freqs_cis, precompute_freqs_cis_3d
from vpp2.models.wan21_14b.wan_video_clip import WanImageEncoderCLIP
from vpp2.models.wan21_14b.wan_video_vae import WanVideoVAE
from vpp2.models.wan21_14b.mot import MoT

LOGGER = logging.getLogger(__name__)
FORMAT = 'vpp2-video-per-row-int8-bf16-v1'
# Official pre_dit reads this layer's weight.device and weight.dtype directly.
# Keeping it BF16 costs only2.6MB and avoids adding a dequantizing weight property.
EXCLUDE_LINEAR = {'time_embedding.0'}


class Int8WeightOnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.register_buffer('qweight', torch.empty((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer('scale', torch.empty((out_features, 1), dtype=torch.float32, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, value):
        if self.scale.dtype != torch.float32 or self.qweight.dtype != torch.int8:
            raise TypeError('INT8 scale must stayFP32; move this model by device only, not dtype')
        # One transient FP32 matrix, then BF16. mul_ avoids a second FP32 matrix.
        weight_fp32 = self.qweight.float()
        weight_fp32.mul_(self.scale)
        weight = weight_fp32.to(dtype=value.dtype)
        del weight_fp32
        return F.linear(value, weight, self.bias)


def replace_linears(model):
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name not in EXCLUDE_LINEAR:
            parent_name, _, attr = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            replacement = Int8WeightOnlyLinear(module.in_features, module.out_features,
                                              module.bias is not None, device='meta')
            setattr(parent, attr, replacement)
            names.append(name)
    return names


def assign_safe(module, path, device):
    expected = set(module.state_dict())
    with safe_open(str(path), framework='pt', device='cpu') as storage:
        actual = set(storage.keys())
        if actual != expected:
            raise ValueError(f'Strict safetensors keys differ: missing={sorted(expected-actual)[:8]}, unexpected={sorted(actual-expected)[:8]}')
        state = {key: storage.get_tensor(key).to(device=device) for key in storage.keys()}
        module.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()


def assign_torch(module, state, *, device, dtype):
    expected = set(module.state_dict())
    if set(state) != expected:
        raise ValueError(f'Strict checkpoint keys differ: missing={sorted(expected-set(state))[:8]}, unexpected={sorted(set(state)-expected)[:8]}')
    converted = {key: value.to(device=device, dtype=dtype) if value.is_floating_point()
                 else value.to(device=device) for key, value in state.items()}
    module.load_state_dict(converted, strict=True, assign=True)
    del converted


class LocalInt8ActionModel(LiberoActionModel):
    def to(self, *args, **kwargs):
        # Base constructor calls self.to(device). Keep auxiliaries on CPU until
        # the exact encoding stage; do not materialize all three on GPU at once.
        if 'dtype' in kwargs or any(isinstance(arg, torch.dtype) for arg in args):
            raise ValueError('This model has fixed INT8/FP32/BF16 mixed storage; .to(dtype) is unsafe')
        self.mot.to(*args, **kwargs)
        if self.proprio_encoder is not None:
            self.proprio_encoder.to(*args, **kwargs)
        return self

    def _restore_auxiliary_component_dtypes(self):
        # The inherited helper moves BOTH encoders toGPU. Each override below
        # instead moves only the encoder it needs and returns it to CPU.
        return None

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, *args, **kwargs):
        try:
            self.vae.to(device=self.device, dtype=torch.float32)
            return super()._encode_input_image_latents_tensor(*args, **kwargs)
        finally:
            self.vae.to(device='cpu', dtype=torch.float32)
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_video_latents(self, *args, **kwargs):
        try:
            self.vae.to(device=self.device, dtype=torch.float32)
            return super()._encode_video_latents(*args, **kwargs)
        finally:
            self.vae.to(device='cpu', dtype=torch.float32)
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _encode_clip_fea_from_video(self, *args, **kwargs):
        try:
            self.clip_encoder.to(device=self.device, dtype=torch.float16)
            return super()._encode_clip_fea_from_video(*args, **kwargs)
        finally:
            self.clip_encoder.to(device='cpu', dtype=torch.float16)
            torch.cuda.empty_cache()


def build_local_int8(root: Path, model_cfg, device):
    from omegaconf import OmegaConf
    root = root.resolve()
    manifest = json.loads((root / 'weights/video_int8_manifest.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Unsupported INT8 artifact format')
    video_cfg = OmegaConf.to_container(model_cfg.video_dit_config, resolve=True)
    action_cfg = OmegaConf.to_container(model_cfg.action_dit_config, resolve=True)
    video_cfg['use_gradient_checkpointing'] = False
    action_cfg['use_gradient_checkpointing'] = False
    with torch.device('meta'):
        video = WanVideoDiT(**video_cfg)
        names = replace_linears(video)
        action = ActionDiT(**action_cfg)
        clip = WanImageEncoderCLIP()
    if names != manifest['quantized_linears']:
        raise ValueError('INT8 artifact architecture/Linear names differ from source config')
    # RoPE caches are plain attributes, not parameters/buffers; recreate after
    # leaving the meta context instead of accidentally keeping meta tensors.
    video.freqs = precompute_freqs_cis_3d(video.attn_head_dim)
    action.freqs = precompute_freqs_cis(action.attn_head_dim, end=1024)
    LOGGER.info('Loading INT8 video directly into%s without BF16/FP32 video allocation', device)
    assign_safe(video, root / 'weights/video_int8.safetensors', device)
    payload = torch.load(root / 'weights/best_action.pt', map_location='cpu', weights_only=True, mmap=True)
    if int(payload.get('step', -1)) != 2250:
        raise ValueError('Expected best action checkpoint2250')
    assign_torch(action, payload['action_expert'], device=device, dtype=torch.bfloat16)
    wan = root / 'weights/Wan2.1-I2V-14B-480P'
    clip_state = torch.load(wan / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                            map_location='cpu', mmap=True, weights_only=True)
    clip_state = {key: value for key, value in clip_state.items()
                  if key.startswith('visual.') or key == 'log_scale'}
    assign_torch(clip, clip_state, device='cpu', dtype=torch.float16)
    del clip_state
    # The complete VAE is small(~0.5GB FP32); retain it without changing any
    # trained graph or decoder API, and move only around the encoding call.
    vae = WanVideoVAE(z_dim=16)
    vae_state = torch.load(wan / 'Wan2.1_VAE.pth', map_location='cpu', mmap=True, weights_only=True)
    if 'model_state' in vae_state:
        vae_state = vae_state['model_state']
    if not any(key.startswith('model.') for key in vae_state):
        vae_state = {'model.'+key: value for key, value in vae_state.items()}
    assign_torch(vae, vae_state, device='cpu', dtype=torch.float32)
    del vae_state
    mot = MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=False)
    vs = model_cfg.video_scheduler
    acs = model_cfg.action_scheduler
    model = LocalInt8ActionModel(
        video_expert=video, action_expert=action, mot=mot, vae=vae,
        clip_encoder=clip, text_encoder=None, tokenizer=None,
        text_dim=4096, proprio_dim=7, device=str(device), torch_dtype=torch.bfloat16,
        video_train_shift=float(vs.train_shift), video_infer_shift=float(vs.infer_shift),
        video_num_train_timesteps=int(vs.num_train_timesteps),
        video_train_sampling_strategy=str(vs.train_sampling_strategy),
        video_train_beta_alpha=float(vs.beta_alpha), video_train_beta_beta=float(vs.beta_beta),
        action_train_shift=float(acs.train_shift), action_infer_shift=float(acs.infer_shift),
        action_num_train_timesteps=int(acs.num_train_timesteps),
        action_chunk_size=1, action_visible_video_frames=3, action_only_first_frame_fast_path=True)
    # Base constructor creates this tiny Linear, then strict assign the trained7D projection.
    assign_torch(model.proprio_encoder, payload['proprio_encoder'], device=device, dtype=torch.bfloat16)
    step = int(payload['step'])
    del payload
    model.eval().requires_grad_(False)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == 'meta':
            raise RuntimeError(f'Unmaterialized meta tensor:{name}')
    gc.collect()
    LOGGER.info('INT8 model ready: video/action onGPU, CLIP/VAE staged onCPU')
    return model, step
