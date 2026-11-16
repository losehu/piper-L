"""Read-only boot checks and orderly UI exits preserve the original failure."""
import json,socket,sys,tempfile,threading,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rlt_piper.preflight import check_can
from rlt_piper.ui_bridge import Endpoint
from rlt_piper.ui_process import Process

class StartupTests(unittest.TestCase):
    def test_down_can_fails_without_model_or_interface_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=Path(folder)/'leader.json';cfg.write_text(json.dumps(dict(adapter_serials=dict(can0='leader'))))
            calls=[]
            def run(argv,**kw):
                calls.append(argv)
                if argv[0]=='udevadm':return SimpleNamespace(returncode=0,stdout='ID_SERIAL='+('leader' if argv[-1].endswith('can0') else 'follower'))
                return SimpleNamespace(returncode=0,stdout=json.dumps([dict(flags=['NOARP'],linkinfo=dict(info_data={}))]))
            with patch('rlt_piper.preflight.subprocess.run',run),self.assertRaisesRegex(RuntimeError,'restore_rlt_can.sh'):
                check_can(dict(adapter_serial='follower'),cfg)
            self.assertFalse(any('set' in call for call in calls))
    def test_orderly_endpoint_close_is_announced_without_stop_latch(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=SimpleNamespace(stop=SimpleNamespace(value=0));endpoint=Endpoint(Path(folder)/'s',guard);endpoint.start()
            conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);conn.settimeout(2);conn.connect(str(endpoint.path));conn.sendall(b'{"action":"heartbeat"}\n')
            first=conn.recv(8192)
            thread=threading.Thread(target=endpoint.close);thread.start();payload=first
            while True:
                data=conn.recv(8192)
                if not data:break
                payload+=data
            conn.close();thread.join(2)
            messages=[json.loads(line) for line in payload.splitlines()]
            self.assertTrue(any(m.get('closing') and not m.get('ready') for m in messages))
            self.assertEqual(guard.stop.value,0);self.assertIsNone(endpoint.error)
    def test_original_startup_error_survives_socket_close_and_is_saved(self):
        package=Path(sys.modules['rlt_piper.ui_bridge'].__file__).parent
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'.venv/bin').mkdir(parents=True);(root/'.venv/bin/python').symlink_to(sys.executable)
            module=root/'rlt_piper';module.mkdir();(module/'__init__.py').write_text('');(module/'ui_bridge.py').symlink_to(package/'ui_bridge.py')
            (module/'cli.py').write_text('''import time,sys
from types import SimpleNamespace
from rlt_piper.ui_bridge import Endpoint
guard=SimpleNamespace(stop=SimpleNamespace(value=0));endpoint=Endpoint.from_environment(guard);endpoint.start()
time.sleep(.2)
print("CAN_BOOT_ERROR: original reason",flush=True)
endpoint.close()
time.sleep(.25)
sys.exit(2)
''')
            process=Process(root);process.start();until=time.monotonic()+5
            while process.active and time.monotonic()<until:process.poll();time.sleep(.01)
            state=process.poll()
            if process.active:process.stop();self.fail('Fake startup did not exit')
            self.assertFalse(process.cancelled,process.take_logs());self.assertEqual(state['phase'],'error');self.assertEqual(state['exit_code'],2)
            self.assertIn('CAN_BOOT_ERROR: original reason',Path(state['controller_log']).read_text())

if __name__=='__main__':unittest.main()
