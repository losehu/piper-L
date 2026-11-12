#!/usr/bin/env python3
"""Loopback-only VPP2 action inference. No robot, CAN, camera, or enable calls.

The caller owns motion scheduling, age limits, command limits and damping.
This server returns predictions; it never executes robot commands.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_REQUEST_BYTES = 4 * 1024 * 1024
PROTOCOL = 'vpp2-piper-v1'


def decode_camera(encoded: str) -> np.ndarray:
    """640x480 JPEG RGB -> training PIL bilinear224x168 +28px black borders."""
    if not isinstance(encoded, str) or len(encoded) > 2_000_000:
        raise ValueError('Camera must be a base64 JPEG of at most 1.5 MB')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError('Invalid camera base64') from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != 'JPEG' or image.size != (640, 480):
                raise ValueError('Camera input must be a 640x480 JPEG; cam0=external, cam1=wrist')
            image = image.convert('RGB')
            resized = image.resize((224, 168), resample=Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (224, 224), 0)
            canvas.paste(resized, (0, 28))
            return np.array(canvas, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Invalid JPEG') from exc


def parse_request(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(payload, dict) or payload.get('protocol') != PROTOCOL:
        raise ValueError(f'protocol must be {PROTOCOL}')
    if payload.get('state_units') != 'rad,m':
        raise ValueError('state_units must be rad,m (six measured joints radians, gripper metres)')
    if payload.get('camera_order') != ['cam0', 'cam1']:
        raise ValueError('camera_order must be [cam0,cam1] (external,wrist)')
    state = np.asarray(payload.get('state'), dtype=np.float32)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('state must contain seven finite measured values')
    # This checks gross unit mistakes. Workspace/slew limits belong to the controller.
    if np.abs(state[:6]).max() > 2 * np.pi or not 0 <= state[6] <= 0.10:
        raise ValueError('Measured state exceeds gross physical bounds; check units')
    return decode_camera(payload.get('cam0_jpeg_b64')), decode_camera(payload.get('cam1_jpeg_b64')), state


class Policy:
    def __init__(self, root: Path, device: str = 'cuda:0', denoising_steps: int = 5, backend: str = 'int8'):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from vpp2.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('This validated deployment requires a CUDA GPU; no CPU fallback')
        torch.cuda.set_device(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU must support BF16')
        if backend == 'int8':
            # Leave display/context/workspace headroom on the24GB collector.
            # The same22GiB allocator cap is also used in the H100 dry run.
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, (22 * 1024**3) / total), self.device)
        self.denoising_steps = int(denoising_steps)
        if self.denoising_steps not in (5,10):
            raise ValueError('Use the selected5-step sampler or the10-step reference')
        root = root.resolve()
        cfg = OmegaConf.load(root / 'config/train_config.yaml')
        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        wan = root / 'weights/Wan2.1-I2V-14B-480P'
        self.checkpoint = root / 'weights/best_action.pt'
        self.video_checkpoint = root / 'weights/video_step010000.pt'
        self.backend = str(backend)
        video_files = [root / 'weights/video_int8.safetensors', root / 'weights/video_int8_manifest.json'] if backend == 'int8' else [self.video_checkpoint]
        for required in (self.checkpoint, *video_files, wan / 'Wan2.1_VAE.pth',
                         wan / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
                         root / 'assets/text_context.pt', root / 'assets/dataset_stats.json'):
            if not required.is_file():
                raise FileNotFoundError(required)
        model_cfg.model_id = str(wan)
        model_cfg.dit_checkpoint_path = str(self.video_checkpoint)
        model_cfg.tokenizer_model_id = str(wan / 'google/umt5-xxl')
        model_cfg.load_text_encoder = False
        model_cfg.load_clip_encoder = True
        model_cfg.action_dit_pretrained_path = None
        model_cfg.redirect_common_files = False
        model_cfg.video_dit_config.use_gradient_checkpointing = False
        model_cfg.action_dit_config.use_gradient_checkpointing = False
        torch.manual_seed(42)
        if backend == 'int8':
            from local_int8 import build_local_int8
            self.model, self.step = build_local_int8(root, model_cfg, self.device)
        elif backend == 'bf16':
            self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=str(self.device))
            self.step = self.model.load_checkpoint(str(self.checkpoint), load_video_from_action_checkpoint=False)
        else:
            raise ValueError('Unknown backend')
        if int(self.step) != 2250:
            raise RuntimeError(f'Expected selected checkpoint2250, got {self.step}')
        self.model.eval().requires_grad_(False)
        stats = json.loads((root / 'assets/dataset_stats.json').read_text())
        provenance = stats.get('provenance', {})
        if not provenance.get('train_only') or not provenance.get('joint_delta_relative_to_first_measured_state'):
            raise ValueError('Expected train-only joint-delta normalization stats')
        def normalizer(field):
            values = {key.removeprefix('global_'): torch.tensor(value, dtype=torch.float32)
                      for key, value in stats[field]['default'].items() if key.startswith('global_')}
            return SingleFieldLinearNormalizer(values, mode='min/max')
        self.state_normalizer = normalizer('state')
        self.action_normalizer = normalizer('action')
        cached = torch.load(root / 'assets/text_context.pt', map_location='cpu', weights_only=True)
        self.context = cached['context'].clone()
        cached_mask = cached['mask'].to(torch.bool)
        if tuple(self.context.shape) != (128, 4096) or tuple(cached_mask.shape) != (128,):
            raise ValueError('Expected exact training prompt context[128,4096] and mask[128]')
        # RobotVideoDataset does this after cache load. Reproduce it exactly.
        self.context[~cached_mask] = 0
        self.context_mask = torch.ones_like(cached_mask)
        self.ready_at = time.time()
        self.lock = threading.Lock()

    def infer_prepared(self, cam0: np.ndarray, cam1: np.ndarray, state: np.ndarray, seed: int = 42) -> dict:
        """Already-letterboxed RGB uint8 images, measured state -> absolute32x7."""
        torch = self.torch
        if cam0.shape != (224, 224, 3) or cam1.shape != (224, 224, 3):
            raise ValueError('Prepared images must both be224x224x3')
        if cam0.dtype != np.uint8 or cam1.dtype != np.uint8:
            raise ValueError('Prepared images must be uint8 RGB')
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('Invalid measured state')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('Another inference is active; controller must retain its damping watchdog')
        try:
            joined = np.concatenate((cam0, cam1), axis=1).copy()
            image = torch.from_numpy(joined).permute(2, 0, 1).float().div(255).sub(.5).div(.5).unsqueeze(0)
            proprio = self.state_normalizer.forward(torch.from_numpy(state.copy()))
            torch.cuda.synchronize(self.device)
            begin = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.model.infer_action(
                    prompt=None, input_image=image, action_horizon=32, proprio=proprio,
                    context=self.context, context_mask=self.context_mask,
                    num_video_frames=17, num_inference_steps=self.denoising_steps, seed=int(seed), tiled=False)
            torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - begin) * 1000
            normalized = output['action'].detach().to('cpu', torch.float32).reshape(32, 7)
            action = self.action_normalizer.backward(normalized).clone()
            action[:, :6] += torch.from_numpy(state[:6]).reshape(1, 6)
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError('Nonfinite action prediction')
            return dict(protocol=PROTOCOL, ok=True, checkpoint_step=int(self.step), backend=self.backend,
                        denoising_steps=self.denoising_steps,
                        actions=action.tolist(), action_units='rad,m', action_type='absolute_joint_targets',
                        action_rate_hz=30, action_horizon=32, inference_ms=inference_ms,
                        normalized_action=normalized.tolist(), seed=int(seed),
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    server_version = 'VPP2PiPER/1'
    def _send(self, status, payload):
        raw = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != '/health':
            return self._send(404, dict(ok=False, error='not_found'))
        return self._send(200, dict(protocol=PROTOCOL, ok=True, model_loaded=True,
                                   checkpoint_step=int(self.server.policy.step),
                                   action_horizon=32,action_rate_hz=30,
                                   warmed_up=getattr(self.server.policy,'warmup_ms',None) is not None,
                                   compiled_action=getattr(self.server.policy,'compiled_action',False),
                                   denoising_steps=self.server.policy.denoising_steps,
                                   warmup_ms=getattr(self.server.policy,'warmup_ms',None),
                                   robot_connected=False, busy=self.server.policy.lock.locked()))

    def do_POST(self):
        request_id = None
        if self.path != '/infer':
            return self._send(404, dict(ok=False, error='not_found'))
        try:
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('Chunked transfer is unsupported')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError('Invalid Content-Length')
            self.connection.settimeout(5)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Truncated request')
            payload = json.loads(raw)
            request_id = payload.get('request_id') if isinstance(payload, dict) else None
            if not isinstance(request_id, (str, int)) or len(str(request_id)) > 128:
                raise ValueError('request_id must be a short string or integer')
            if payload.get('seed', 42) != 42:
                raise ValueError('This deployment uses fixed seed42 as validated')
            camera0, camera1, state = parse_request(payload)
            start = time.perf_counter()
            result = self.server.policy.infer_prepared(camera0, camera1, state, seed=42)
            # normalized_action is useful for offline tests, never for the hardware client.
            result.pop('normalized_action')
            result.update(request_id=request_id, server_total_ms=(time.perf_counter() - start) * 1000)
            self._send(200, result)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, dict(protocol=PROTOCOL, ok=False, request_id=request_id, error=str(exc)))
        except BlockingIOError as exc:
            self._send(409, dict(protocol=PROTOCOL, ok=False, request_id=request_id, error=str(exc)))
        except Exception:
            logging.exception('Inference failed')
            self._send(503, dict(protocol=PROTOCOL, ok=False, request_id=request_id, error='inference_failed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backend', choices=('int8', 'bf16'), default='int8')
    parser.add_argument('--host', choices=('127.0.0.1',), default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--warmup-sample',type=Path)
    parser.add_argument('--compile-action',action='store_true')
    parser.add_argument('--denoising-steps',type=int,choices=[5,10],default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch_threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    import torch
    torch.set_num_threads(torch_threads)
    policy = Policy(args.root, args.device,denoising_steps=args.denoising_steps,backend=args.backend)
    policy.compiled_action=bool(args.compile_action)
    if args.compile_action:
        policy.model.mot.forward_action_with_video_cache=torch.compile(
            policy.model.mot.forward_action_with_video_cache,fullgraph=False,dynamic=False,
            options={'triton.cudagraphs':False,'compile_threads':2})
        logging.info('torch.compile enabled for action denoising; CUDA graphs disabled')
    if args.warmup_sample:
        with np.load(args.warmup_sample,allow_pickle=False) as sample:
            cameras=[]
            for key in ['cam0','cam1']:
                image=Image.fromarray(np.asarray(sample[key],dtype=np.uint8)).convert('RGB')
                if image.size!=(224,224):
                    if image.size!=(640,480):raise ValueError('Unexpected warmup camera dimensions')
                    resized=image.resize((224,168),resample=Image.Resampling.BILINEAR)
                    image=Image.new('RGB',(224,224),0);image.paste(resized,(0,28))
                cameras.append(np.array(image))
            state=np.array(sample['state'],dtype=np.float32)
        result=policy.infer_prepared(*cameras,state,seed=42)
        policy.warmup_ms=result['inference_ms']
        logging.info('Saved-observation CUDA warmup complete in%.1fms; no robot connected',policy.warmup_ms)
    service = ThreadingHTTPServer((args.host, args.port), Handler)
    service.daemon_threads = True
    service.policy = policy
    logging.info('Ready http://%s:%d checkpoint%s; no robot connected', args.host, args.port, policy.step)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.server_close()


if __name__ == '__main__':
    main()
