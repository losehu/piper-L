"""Control-process RLT client; no GPU or Corrector dependency."""
import uuid,numpy as np
from rlt_piper.contract import SCHEMA

class Client:
    def __init__(self,stage):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        self.codec=msgpack_numpy;self.session=uuid.uuid4().hex;self.counter=0
        self.ws=connect('ws://127.0.0.1:8003',proxy=None,compression=None,open_timeout=3,close_timeout=.2,ping_interval=None,max_size=8*1024**2)
        try:
            self.metadata=self.codec.unpackb(self.ws.recv(timeout=3));m=self.metadata
            if (m.get('protocol')!=SCHEMA or m.get('base_step')!=2500 or m.get('rlt_stage')!=stage
                    or m.get('action_horizon')!=50 or m.get('physical_action_dim')!=7
                    or m.get('corrector_enabled') is not False or not m.get('warmed_up')
                    or m.get('robot_connected') is not False):raise RuntimeError('RLT service identity mismatch')
        except BaseException:self.ws.close();raise
    def request(self,method,obs=None,command=None,generation=0,prompt=None):
        self.counter+=1;payload=dict(method=method,session=self.session,request_id=self.counter,generation=int(generation))
        if obs is not None:
            from openpi_client.image_tools import resize_with_pad
            payload.update(state=np.asarray(obs.state,np.float32),observed_at=float(obs.t),prompt=prompt,
                           **{k:resize_with_pad(v,224,224) for k,v in obs.images.items()})
        self.ws.send(self.codec.packb(payload));raw=self.ws.recv(timeout=2)
        if isinstance(raw,str):raise RuntimeError(raw)
        result=self.codec.unpackb(raw)
        if (result.get('protocol')!=SCHEMA or result.get('request_id')!=self.counter or result.get('session')!=self.session
                or result.get('ok') is not True or result.get('generation')!=generation):raise RuntimeError('RLT response identity mismatch')
        if obs is not None and abs(float(result.get('observed_at',float('inf')))-obs.t)>1e-6:
            raise RuntimeError('RLT response belongs to another observation')
        return result
    def close(self):self.ws.close()
