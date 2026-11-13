"""Small control-process client; imports no model/GPU libraries."""
import threading
import uuid
import numpy as np
from corrector.contract import VERSION


class Session:
    def __init__(self,mode):
        if mode not in ('monitor','truncate','full'):raise ValueError('Invalid Corrector mode')
        self.id=uuid.uuid4().hex;self.mode=mode;self.counter=0;self.lock=threading.Lock()

    def next(self):
        with self.lock:self.counter+=1;return self.counter


class Client:
    def __init__(self,port,session):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        self.codec=msgpack_numpy;self.session=session
        # This service is strictly local. websockets15 otherwise discovers
        # desktop proxy settings, which can fail before cold startup is tried.
        self.ws=connect('ws://127.0.0.1:%d'%port,proxy=None,compression=None,max_size=8*1024**2,open_timeout=3,close_timeout=.2,ping_interval=None)
        self.metadata=self.codec.unpackb(self.ws.recv(timeout=3))
        if (self.metadata.get('protocol')!=VERSION or self.metadata.get('training_updates')!=2500
            or self.metadata.get('source_checkpoint_index')!=2499 or self.metadata.get('action_horizon')!=50
            or self.metadata.get('physical_action_dim')!=7 or not self.metadata.get('warmed_up')
            or self.metadata.get('robot_connected') is not False):
            self.close();raise RuntimeError('Wrong pi05 Corrector service identity')

    def request(self,method,obs=None,command=None,generation=0,prompt=None):
        request_id=self.session.next()
        payload=dict(method=method,request_id=request_id,session=self.session.id,mode=self.session.mode,generation=generation)
        if obs is not None:
            from openpi_client.image_tools import resize_with_pad
            payload.update(state=obs.state.astype(np.float32),observed_at=float(obs.t),prompt=prompt,
                executed_command=np.asarray(command,dtype=np.float32))
            payload.update({k:resize_with_pad(image,224,224) for k,image in obs.images.items()})
        self.ws.send(self.codec.packb(payload));raw=self.ws.recv(timeout=2)
        if isinstance(raw,str):raise RuntimeError(raw)
        result=self.codec.unpackb(raw)
        if (result.get('protocol')!=VERSION or result.get('request_id')!=request_id
            or result.get('session')!=self.session.id or result.get('mode')!=self.session.mode or result.get('ok') is not True):
            raise RuntimeError('Corrector response request/session mismatch')
        return result

    def close(self):self.ws.close()
