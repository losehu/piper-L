"""UI command provenance, disconnect fail-stop and model-free demo checks."""
import json,socket,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from rlt_piper.ui_bridge import Endpoint
from rlt_piper.hil import Handover
from rlt_piper.ui_process import Demo

class UiTests(unittest.TestCase):
    def endpoint(self,path):
        guard=SimpleNamespace(stop=SimpleNamespace(value=0));e=Endpoint(path,guard);h=Handover();labels=[]
        def outcome(value,g):labels.append(value);g.stop.value=1
        e.attach(SimpleNamespace(outcome=outcome),h,None);e.update(ready=True,phase='running');return e,guard,h,labels
    def test_directional_commands_reject_stale_generation_and_double_toggle(self):
        e,guard,h,labels=self.endpoint('/unused')
        with self.assertRaises(ValueError):e.command(dict(action='takeover',generation=9))
        self.assertFalse(h.requested)
        e.command(dict(action='takeover',generation=0));self.assertTrue(h.requested)
        with self.assertRaises(ValueError):e.command(dict(action='takeover',generation=0))
        self.assertEqual(labels,[]);self.assertEqual(guard.stop.value,0)
    def test_results_and_stop_use_existing_guard(self):
        e,guard,h,labels=self.endpoint('/unused');e.command(dict(action='success'));self.assertEqual(labels,[1]);self.assertTrue(guard.stop.value)
        e,guard,h,labels=self.endpoint('/unused');e.command(dict(action='stop'));self.assertTrue(guard.stop.value);self.assertEqual(labels,[])
    def test_preparation_rejects_outcome_and_takeover_but_accepts_stop(self):
        e,guard,h,labels=self.endpoint('/unused');e.update(ready=False)
        for action in ['success','failure','takeover','resume']:
            with self.assertRaises(ValueError):e.command(dict(action=action))
        e.command(dict(action='stop'));self.assertTrue(guard.stop.value);self.assertFalse(h.requested)
    def test_ui_disconnect_requests_damping_guard_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=SimpleNamespace(stop=SimpleNamespace(value=0));e=Endpoint(Path(folder)/'s',guard);e.start()
            conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);conn.connect(str(e.path));conn.sendall(b'{"action":"heartbeat"}\n');conn.close()
            until=time.monotonic()+2
            while not guard.stop.value and time.monotonic()<until:time.sleep(.01)
            e.close();self.assertTrue(guard.stop.value);self.assertIsNotNone(e.error)
    def test_lost_heartbeat_stops_without_ui_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=SimpleNamespace(stop=SimpleNamespace(value=0));e=Endpoint(Path(folder)/'s',guard);e.start()
            conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);conn.connect(str(e.path))
            until=time.monotonic()+2
            while not guard.stop.value and time.monotonic()<until:time.sleep(.02)
            conn.close();e.close();self.assertTrue(guard.stop.value)
    def test_preview_reads_existing_bank_and_preserves_image_ratio(self):
        import base64,io,numpy as np
        from PIL import Image
        e,guard,h,labels=self.endpoint('/unused')
        e.bank=SimpleNamespace(observation=lambda:SimpleNamespace(t=10.,images={'cam0':np.zeros((480,640,3),np.uint8)}))
        preview=e.preview();image=Image.open(io.BytesIO(base64.b64decode(preview['images']['cam0'])))
        self.assertEqual(image.size,(300,225));self.assertEqual(preview['camera_capture_t'],10.)
    def test_demo_never_constructs_controller_and_can_complete_workflow(self):
        demo=Demo('/unused');demo.start();demo.send('takeover');self.assertEqual(demo.state['mode'],'human')
        demo.send('resume');self.assertEqual(demo.state['mode'],'policy');demo.send('failure')
        self.assertEqual(demo.state['outcome'],0);self.assertFalse(demo.active)

if __name__=='__main__':unittest.main()
