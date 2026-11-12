"""Wire/persistent HTTP CPU tests with fake policy; no CUDA/CAN/cameras."""
import json
import queue
import struct
import threading
import time
import unittest
import os
from unittest import mock
from types import SimpleNamespace
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .server import Handler
from .transport import Connection,PolicyClient
from .wire import MODES,PROTOCOL,MAGIC,jpeg_pair,prepare_jpegs,pixel_hash,encode,decode


def images():
    y,x=np.indices((480,640));a=np.stack([x%256,y%256,(x+y)%256],axis=-1).astype(np.uint8)
    return dict(cam0=a,cam1=a[:,::-1].copy())


def prediction():
    return dict(ok=True,protocol=PROTOCOL,checkpoint_step=2250,denoising_steps=5,
        actual_denoising_calls=5,actions=np.tile([.1,.2,-.3,.4,.5,-.6,.02],(32,1)).tolist(),
        action_units='rad,m',action_type='absolute_joint_targets',action_rate_hz=30,
        action_horizon=32,inference_ms=1.)


@contextmanager
def fake_server():
    class QuietHandler(Handler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),QuietHandler);server.daemon_threads=True
    server.counter=0;server.counter_lock=threading.Lock();server.jobs=queue.Queue()
    server.echo_result=prediction();server.infers=0;server.override={}
    def infer_loop():
        while True:
            job=server.jobs.get()
            if job is None:return
            server.infers+=1;time.sleep(getattr(server,'delay',0))
            job.result=dict(prediction(),**server.override);job.done.set()
    worker=threading.Thread(target=infer_loop,daemon=True);worker.start()
    http=threading.Thread(target=server.serve_forever,daemon=True);http.start()
    try:yield server
    finally:server.shutdown();server.server_close();server.jobs.put(None);worker.join(1);http.join(1)


class WireTests(unittest.TestCase):
    def test_every_codec_is_pixel_identical_and_keeps_camera_order(self):
        source=images();state=np.array([0,.2,-.2,0,.1,0,.02]);wanted=prepare_jpegs(jpeg_pair(source))
        hashes=[]
        for mode in MODES:
            raw,kind=encode(source,state,'unit:'+mode,mode)
            a,b,actual,meta,codec,pixels=decode(raw,kind)
            np.testing.assert_array_equal(a,wanted[0]);np.testing.assert_array_equal(b,wanted[1])
            np.testing.assert_allclose(state,actual);hashes.append(pixels)
        self.assertEqual(len(set(hashes)),1)

    def test_truncation_extra_data_and_wrong_pixel_checksum_are_rejected(self):
        raw,kind=encode(images(),np.r_[np.zeros(6),.02],'unit','png_prepared')
        for invalid in [raw[:-1],raw+b'garbage']:
            with self.assertRaises(ValueError):decode(invalid,kind)
        n=struct.unpack('!I',raw[4:8])[0];meta=json.loads(raw[8:8+n]);meta['pixel_sha256']='0'*64
        header=json.dumps(meta,separators=(',',':')).encode()
        altered=MAGIC+struct.pack('!I',len(header))+header+raw[8+n:]
        with self.assertRaisesRegex(ValueError,'pixels changed'):decode(altered,kind)


class HttpTests(unittest.TestCase):
    def test_persistent_connection_is_reused_and_actions_remain_absolute(self):
        with fake_server() as server:
            client=Connection(server.server_port,'webp_prepared');client.health();ids=[]
            for _ in range(3):
                actions,row,body,kind=client.infer(images(),np.r_[np.zeros(6),.02])
                ids.append(row['connection_id']);np.testing.assert_allclose(actions[0],[.1,.2,-.3,.4,.5,-.6,.02])
            echo=client.echo(body,kind);self.assertEqual(echo['connection_id'],ids[-1]);client.close()
            self.assertEqual(len(set(ids)),1);self.assertEqual(server.infers,3)

    def test_close_mode_uses_new_connections(self):
        with fake_server() as server:
            client=Connection(server.server_port,'json_close');client.health()
            ids=[client.infer(images(),np.r_[np.zeros(6),.02])[1]['connection_id'] for _ in range(3)]
            self.assertEqual(len(set(ids)),3);client.close()

    def test_closed_client_does_not_send_again(self):
        with fake_server() as server:
            client=Connection(server.server_port);client.health();client.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):client.infer(images(),np.r_[np.zeros(6),.02])
            self.assertEqual(server.infers,0)

    def test_wrong_model_and_action_semantics_are_rejected(self):
        with fake_server() as server:
            client=Connection(server.server_port);client.health()
            for invalid in [dict(checkpoint_step=2500),dict(action_type='delta')]:
                server.override=invalid
                with self.assertRaisesRegex(RuntimeError,'identity'):client.infer(images(),np.r_[np.zeros(6),.02])
            client.close()

    def test_invalid_binary_body_does_not_reach_policy(self):
        with fake_server() as server:
            client=Connection(server.server_port);client.health()
            with self.assertRaisesRegex(RuntimeError,'HTTP400'):
                client._request('/infer',b'broken','application/vnd.piper.vpp2-observation')
            self.assertEqual(server.infers,0);client.close()

    def test_vpp2_runner_can_opt_in_without_changing_absolute_actions(self):
        from rtc_piper.vpp2_client import Vpp2Client,PROMPT
        with fake_server() as server,mock.patch.dict(os.environ,{'VPP2_REMOTE_PORT':str(server.server_port),'VPP2_REMOTE_CODEC':'webp_prepared'}):
            client=Vpp2Client(dict(compile_vpp2=True,denoising_steps=5,request_timeout_s=2))
            observation=SimpleNamespace(images=images(),state=np.r_[np.full(6,.3),.02])
            actions=client.infer(observation,PROMPT)
            np.testing.assert_allclose(actions[0],[.1,.2,-.3,.4,.5,-.6,.02])
            self.assertEqual(client.metadata['inference_location'],'H100');client.close()

    def test_remote_outage_is_fail_fast_not_a_local_model_start(self):
        import socket
        from rtc_piper.vpp2_client import Vpp2Client
        with socket.socket() as reserved:
            reserved.bind(('127.0.0.1',0));port=reserved.getsockname()[1]
            with mock.patch.dict(os.environ,{'VPP2_REMOTE_PORT':str(port)}):
                with self.assertRaisesRegex(RuntimeError,'H100 VPP2 unavailable'):
                    Vpp2Client(dict(compile_vpp2=True,denoising_steps=5,request_timeout_s=.2))

    def test_policy_pool_uses_two_independent_persistent_connections(self):
        from rtc_piper.vpp2_client import PROMPT
        with fake_server() as server:
            server.delay=.03
            client=PolicyClient(server.server_port,dict(request_timeout_s=2,vpp2_max_inflight=2))
            barrier=threading.Barrier(2)
            def request():
                barrier.wait()
                return client.infer(SimpleNamespace(images=images(),state=np.r_[np.zeros(6),.02]),PROMPT)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=[f.result() for f in [pool.submit(request),pool.submit(request)]]
            self.assertEqual([c.counter for c in client.connections],[1,1])
            self.assertEqual(client.metadata['max_inflight'],2)
            np.testing.assert_array_equal(results[0],results[1]);client.close()


if __name__=='__main__':unittest.main()
