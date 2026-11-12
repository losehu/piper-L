"""Strict loopback adapter for the trained2250 VPP2's absolute32x7 protocol."""
import base64
import io
import json
import os
import http.client
import threading
import urllib.request
import uuid
import numpy as np
from PIL import Image

PROTOCOL='vpp2-piper-v1'
PROMPT='Pick up all the objects on the table and place them in the box.'


class Vpp2Client:
    def __init__(self,settings):
        self.remote=None
        remote_port=os.environ.get('VPP2_REMOTE_PORT') or settings.get('vpp2_remote_port')
        if remote_port:
            if not settings.get('compile_vpp2') or settings['denoising_steps']!=5:
                raise ValueError('Remote H100 VPP2 requires5 steps and --compile')
            from vpp2_network.transport import PolicyClient
            try:
                self.remote=PolicyClient(int(remote_port),settings,
                    os.environ.get('VPP2_REMOTE_CODEC','webp_prepared'))
            except (OSError,http.client.HTTPException) as error:
                # ensure_policy must not fall back to the local GPU on outage.
                raise RuntimeError('H100 VPP2 unavailable; start the H100 bridge first: '+str(error)) from error
            self.metadata=self.remote.metadata
            print('VPP2：H100单卡BF16，5步编译推理，HTTP长连接。',flush=True)
            return
        self.settings=settings
        self.url='http://127.0.0.1:%d'%settings['policy_port']
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.closed=threading.Event()
        self.session=uuid.uuid4().hex
        self.request_id=0
        with self.opener.open(self.url+'/health',timeout=2) as response:health=json.load(response)
        if (health.get('protocol')!=PROTOCOL or health.get('ok') is not True
                or health.get('model_loaded') is not True or health.get('checkpoint_step')!=2250
                or health.get('warmed_up') is not True
                or bool(health.get('compiled_action',False))!=settings.get('compile_vpp2',False)
                or health.get('denoising_steps')!=settings['denoising_steps']
                or health.get('robot_connected') is not False):
            raise RuntimeError('Wrong VPP2 model identity or service not ready')
        self.metadata=dict(health,model='vpp2_piper_2250',action_horizon=32,
            physical_action_dim=7,fps=30,action_semantics='absolute joint radians and gripper metres',
            execution_mode='plan_execute',checkpoint=settings.get('checkpoint','weights/best_action.pt'))

    def infer(self,observation,prompt):
        if self.remote is not None:return self.remote.infer(observation,prompt)
        if self.closed.is_set():raise RuntimeError('VPP2 client closed')
        if prompt!=PROMPT:raise ValueError('VPP2 uses the fixed trained task text cache')
        self.request_id+=1;request_id='%s:%d'%(self.session,self.request_id)
        payload=dict(protocol=PROTOCOL,request_id=request_id,state_units='rad,m',
            camera_order=['cam0','cam1'],state=np.asarray(observation.state,dtype=float).tolist(),seed=42)
        for key in ['cam0','cam1']:
            image=np.asarray(observation.images[key])
            if image.shape!=(480,640,3) or image.dtype!=np.uint8:
                raise ValueError('VPP2 camera must be raw640x480 RGB uint8')
            buffer=io.BytesIO()
            Image.fromarray(image).save(buffer,format='JPEG',quality=95)
            payload[key+'_jpeg_b64']=base64.b64encode(buffer.getvalue()).decode('ascii')
        request=urllib.request.Request(self.url+'/infer',data=json.dumps(payload,allow_nan=False).encode(),
            headers={'Content-Type':'application/json'})
        with self.opener.open(request,timeout=self.settings['request_timeout_s']) as response:result=json.load(response)
        if self.closed.is_set():raise RuntimeError('Late VPP2 result after client cancellation')
        actions=np.asarray(result.get('actions'),dtype=float)
        if (result.get('protocol')!=PROTOCOL or result.get('ok') is not True
                or result.get('request_id')!=request_id or result.get('checkpoint_step')!=2250
                or result.get('denoising_steps')!=self.settings['denoising_steps']
                or result.get('action_units')!='rad,m' or result.get('action_type')!='absolute_joint_targets'
                or result.get('action_rate_hz')!=30 or result.get('action_horizon')!=32
                or actions.shape!=(32,7) or not np.isfinite(actions).all()):
            raise RuntimeError('VPP2 response identity/units/rate/action shape mismatch')
        # The VPP2 service already undoes normalization and adds measured joints.
        # Do NOT add observation.state a second time.
        return actions

    def close(self):
        if self.remote is not None:self.remote.close()
        else:self.closed.set()
