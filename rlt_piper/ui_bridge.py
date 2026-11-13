"""Private local UI channel. A lost UI heartbeat requests the existing guard stop."""
import base64,json,os,socket,stat,threading,time
from pathlib import Path

class Endpoint:
    def __init__(self,path,guard):
        self.path=Path(path);self.guard=guard;self.lock=threading.Lock();self.stop=threading.Event()
        self.data=dict(phase='loading',mode='policy',source='hold',learner={},ready=False)
        self.recorder=self.handover=self.bank=None;self.error=None
    @classmethod
    def from_environment(cls,guard):
        path=os.environ.get('PIPER_RLT_UI_SOCKET')
        if not path:return None
        return cls(path,guard)
    def start(self):
        parent=self.path.parent.stat()
        if parent.st_uid!=os.getuid() or stat.S_IMODE(parent.st_mode)!=0o700:raise RuntimeError('UI socket directory must be private')
        self.server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.server.bind(str(self.path));self.path.chmod(0o600)
        self.server.listen(1);self.server.settimeout(.1)
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def update(self,**values):
        with self.lock:self.data.update(values)
    def attach(self,recorder,handover,bank):self.recorder,self.handover,self.bank=recorder,handover,bank
    def command(self,item):
        action=item.get('action')
        if action=='heartbeat':return
        if action=='stop':self.guard.stop.value=1;return
        if action not in ('takeover','resume','success','failure'):raise ValueError('Unknown UI action')
        if not self.data.get('ready') or self.guard.stop.value:raise ValueError('控制尚未就绪或已经停止')
        if action in ('success','failure'):
            self.recorder.outcome(int(action=='success'),self.guard);return
        h=self.handover
        with h.lock:
            if h.requested:raise ValueError('正在切换控制权，请等待')
            if item.get('generation')!=h.generation:raise ValueError('界面状态已更新，请重试')
            if (action=='takeover')!=(h.mode=='policy'):raise ValueError('控制模式已改变')
            h.toggle()
    def preview(self):
        if self.bank is None:return {}
        try:
            import io
            from PIL import Image
            obs=self.bank.observation();result={}
            for key,array in obs.images.items():
                img=Image.fromarray(array);img.thumbnail((400,225));buffer=io.BytesIO();img.save(buffer,format='PNG')
                result[key]=base64.b64encode(buffer.getvalue()).decode('ascii')
            return dict(images=result,camera_capture_t=obs.t)
        except Exception:return {}
    def run(self):
        conn=None;began=time.monotonic();last=began;pending=b'';next_status=next_preview=0;preview={}
        try:
            while not self.stop.is_set():
                if conn is None:
                    if time.monotonic()-began>5:raise RuntimeError('UI did not connect')
                    try:conn,_=self.server.accept();conn.settimeout(.05);last=time.monotonic()
                    except socket.timeout:continue
                try:
                    raw=conn.recv(8192)
                    if not raw:raise RuntimeError('UI disconnected')
                    pending+=raw
                    if len(pending)>65536:raise RuntimeError('UI command too large')
                    while b'\n' in pending:
                        line,pending=pending.split(b'\n',1);item=json.loads(line);last=time.monotonic()
                        try:self.command(item)
                        except ValueError as e:self.update(notice=str(e))
                except socket.timeout:pass
                now=time.monotonic()
                if now-last>1:raise RuntimeError('UI heartbeat expired')
                if now>=next_status:
                    if now>=next_preview:preview=self.preview();next_preview=now+1
                    with self.lock:message=dict(self.data,**preview,t=now,stopped=bool(self.guard.stop.value))
                    if self.handover:
                        with self.handover.lock:message.update(mode=self.handover.mode,generation=self.handover.generation,switching=self.handover.requested)
                    if self.recorder:
                        with self.recorder.lock:message.update(recorded_commands=len(self.recorder.commands),accepted_plans=len(self.recorder.plans),outcome=self.recorder.label)
                    conn.sendall((json.dumps(message,separators=(',',':'))+'\n').encode());next_status=now+.2
        except BaseException as e:
            if not self.stop.is_set():self.error=repr(e);self.guard.stop.value=1
        finally:
            if conn:conn.close()
    def close(self):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join(2)
        if hasattr(self,'server'):self.server.close()
        self.path.unlink(missing_ok=True)
