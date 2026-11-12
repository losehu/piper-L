"""Serial persistent HTTP client, with strict model/action identity checks."""
import http.client
import json
import queue
import socket
import threading
import time
import uuid
import numpy as np
from .wire import encode,PROTOCOL,PREPROCESS


class Connection:
    def __init__(self,port=21969,mode='webp_prepared',timeout=60):
        self.port=int(port);self.mode=mode;self.timeout=timeout
        if not 1024<=self.port<=65535:raise ValueError('Invalid loopback inference port')
        self.keepalive=mode!='json_close';self.session=uuid.uuid4().hex;self.counter=0
        self.connection=None;self.closed=False;self.lock=threading.Lock()

    def _request(self,path,body=None,kind=None):
        if self.closed:raise RuntimeError('Remote inference client closed')
        if self.connection is None:self.connection=http.client.HTTPConnection('127.0.0.1',self.port,timeout=self.timeout)
        connection=self.connection
        headers={'Connection':'keep-alive' if self.keepalive else 'close'}
        if kind:headers['Content-Type']=kind
        start=time.perf_counter()
        try:
            connection.request('GET' if body is None else 'POST',path,body=body,headers=headers)
            response=connection.getresponse();raw=response.read()
            elapsed=1000*(time.perf_counter()-start)
            result=json.loads(raw)
            if response.status!=200 or result.get('ok') is not True:
                raise RuntimeError('Remote HTTP%d: %s'%(response.status,result.get('error','invalid_response')))
            return result,elapsed,len(raw)
        except BaseException:
            # A failed request is never replayed with an old observation.
            connection.close()
            if self.connection is connection:self.connection=None
            raise
        finally:
            if not self.keepalive:
                connection.close()
                if self.connection is connection:self.connection=None

    def health(self):
        with self.lock:result,_,_=self._request('/health')
        if (result.get('protocol')!=PROTOCOL or result.get('checkpoint_step')!=2250
            or result.get('denoising_steps')!=5 or result.get('compiled_action') is not True
            or result.get('warmed_up') is not True or result.get('model_gpu_resident') is not True
            or result.get('backend')!='bf16' or result.get('preprocess')!=PREPROCESS
            or result.get('robot_connected') is not False):raise RuntimeError('Remote model identity mismatch')
        return result

    def infer(self,images,state,expected_pixels=None):
        before=time.perf_counter()
        with self.lock:
            if self.closed:raise RuntimeError('Remote inference client closed')
            self.counter+=1;request_id='%s:%d'%(self.session,self.counter)
            body,kind=encode(images,state,request_id,self.mode)
            encoded=time.perf_counter()
            result,http_ms,response_bytes=self._request('/infer',body,kind)
        if self.closed:raise RuntimeError('Late response after remote client closed')
        actions=np.asarray(result.get('actions'),dtype=float)
        if (result.get('protocol')!=PROTOCOL or result.get('request_id')!=request_id
            or result.get('checkpoint_step')!=2250 or result.get('denoising_steps')!=5
            or result.get('actual_denoising_calls')!=5 or result.get('action_units')!='rad,m'
            or result.get('action_type')!='absolute_joint_targets' or result.get('action_rate_hz')!=30
            or result.get('action_horizon')!=32 or result.get('preprocess')!=PREPROCESS
            or result.get('benchmark_echo_only') is True or actions.shape!=(32,7)
            or not np.isfinite(actions).all()):raise RuntimeError('Remote action identity/shape/units mismatch')
        if expected_pixels is not None and result.get('pixel_sha256')!=expected_pixels:raise RuntimeError('Model input pixels changed')
        row=dict(total_ms=1000*(time.perf_counter()-before),encode_ms=1000*(encoded-before),http_ms=http_ms,
            inference_ms=result['inference_ms'],server_total_ms=result['server_total_ms'],
            server_decode_ms=result['server_decode_ms'],body_read_ms=result['server_body_read_ms'],
            connection_id=result['connection_id'],request_bytes=len(body),response_bytes=response_bytes,
            pixel_sha256=result['pixel_sha256'],actual_denoising_calls=result['actual_denoising_calls'])
        return actions,row,body,kind

    def echo(self,body,kind):
        with self.lock:result,elapsed,size=self._request('/echo',body,kind)
        if result.get('benchmark_echo_only') is not True:raise RuntimeError('Unexpected echo result')
        return dict(total_ms=elapsed,server_total_ms=result['server_total_ms'],connection_id=result['connection_id'],
            request_bytes=len(body),response_bytes=size)

    def close(self):
        self.closed=True
        if self.connection is not None:
            if self.connection.sock is not None:
                try:self.connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:pass
            self.connection.close();self.connection=None


class PolicyClient:
    """Drop-in observation/action adapter for the existing synchronous runner."""
    def __init__(self,port,settings,mode='webp_prepared'):
        self.connections=[Connection(port,mode,timeout=settings['request_timeout_s'])
            for _ in range(int(settings.get('vpp2_max_inflight',1)))]
        if not 1<=len(self.connections)<=2:raise ValueError('Use one or two persistent inference connections')
        self.connection=self.connections[0];self.pool=queue.Queue();self.closed=False
        try:
            for connection in self.connections:health=connection.health();self.pool.put(connection)
        except BaseException:
            for connection in self.connections:connection.close()
            raise
        self.metadata=dict(health,model='vpp2_piper_2250',physical_action_dim=7,fps=30,
            execution_mode=settings.get('execution_mode','h100_async_rtc'),action_semantics='absolute joint radians and gripper metres',
            inference_location='H100',transport_mode=mode,checkpoint='weights/best_action.pt',
            max_inflight=len(self.connections))
        self.last_stats=None

    def infer(self,observation,prompt):
        from rtc_piper.vpp2_client import PROMPT
        if prompt!=PROMPT:raise ValueError('VPP2 uses the fixed trained task text')
        if self.closed:raise RuntimeError('Remote policy client closed')
        connection=self.pool.get()
        try:
            if self.closed:raise RuntimeError('Remote policy client closed')
            actions,self.last_stats,_,_=connection.infer(observation.images,observation.state)
            return actions
        finally:self.pool.put(connection)

    def close(self):
        self.closed=True
        for connection in self.connections:connection.close()
