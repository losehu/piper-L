"""UI-owned subprocess/PTY and private IPC. Imports no robot or ML packages."""
import codecs,fcntl,json,os,pty,signal,socket,subprocess,tempfile,termios,time
from pathlib import Path

class Process:
    def __init__(self,root):
        self.root=Path(root);self.process=None;self.state=dict(phase='idle',mode='policy',learner={})
        self.conn=None;self.directory=None;self.master=None;self.pending=b'';self.lines=[];self.cancelled=False
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace');self.next_heartbeat=0;self.command_kind=None
        self.policy_log=None
    @property
    def active(self):return self.process is not None and self.process.poll() is None
    def start(self,stage='online',learn=True,check=False):
        if self.active:raise RuntimeError('已有本窗口启动的任务在运行')
        self.cleanup();self.state=dict(phase='checking' if check else 'loading',mode='policy',learner={},ready=False)
        self.cancelled=False;self.command_kind='check' if check else 'run';self.pending=b'';self.lines=[]
        self.decoder=codecs.getincrementaldecoder('utf-8')('replace')
        self.directory=tempfile.TemporaryDirectory(prefix='piper_rlt_ui_');self.path=Path(self.directory.name)/'control.sock'
        argv=[str(self.root/'.venv/bin/python'),'-u','-m','rlt_piper.cli',self.command_kind,'--stage','collect' if check else stage]
        if not learn:argv.append('--no-train')
        env=dict(os.environ,PIPER_DEPLOY_ROOT=str(self.root),PYTHONPATH=str(self.root))
        env.pop('PIPER_RLT_UI_SOCKET',None)
        if not check:env['PIPER_RLT_UI_SOCKET']=str(self.path)
        master,slave=pty.openpty();self.master=master;os.set_blocking(master,False)
        def session():
            os.setsid();fcntl.ioctl(0,termios.TIOCSCTTY,0)
        try:self.process=subprocess.Popen(argv,cwd=self.root,env=env,stdin=slave,stdout=slave,stderr=slave,preexec_fn=session,close_fds=True)
        except BaseException:self.cleanup();raise
        finally:os.close(slave)
        try:self.policy_log=(self.root/'logs/rlt_policy.log').open();self.policy_log.seek(0,2)
        except FileNotFoundError:self.policy_log=None
        self.lines.append('仅检查模型，不打开机械臂。\n' if check else '正在启动：加载模型 → 准备机械臂 → 自动对齐 → 执行并记录。\n')
    def send(self,action):
        if not self.conn:raise RuntimeError('控制连接尚未就绪')
        packet=dict(action=action,generation=self.state.get('generation',0))
        self.conn.sendall((json.dumps(packet)+'\n').encode())
    def stop(self):
        if not self.active:return
        self.cancelled=True;self.state.update(phase='stopping',ready=False)
        try:
            if self.conn:self.send('stop')
        except OSError:pass
        finally:
            try:os.killpg(self.process.pid,signal.SIGINT)
            except ProcessLookupError:pass
    def poll(self):
        if self.process is None:return self.state
        if self.master is not None:
            try:
                for _ in range(16):
                    data=os.read(self.master,16384)
                    if not data:break
                    self.lines.append(self.decoder.decode(data).replace('\r',''))
            except (BlockingIOError,OSError):pass
        if self.policy_log:
            text=self.policy_log.read(32768)
            if text:self.lines.append(text)
        if self.active and self.command_kind=='run' and not self.cancelled:
            try:
                if self.conn is None and self.path.exists():
                    self.conn=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.conn.connect(str(self.path));self.conn.setblocking(False)
                if self.conn:
                    if time.monotonic()>=self.next_heartbeat:self.send('heartbeat');self.next_heartbeat=time.monotonic()+.2
                    for _ in range(32):
                        try:data=self.conn.recv(65536)
                        except BlockingIOError:break
                        if not data:raise OSError('控制连接断开')
                        self.pending+=data
                        if len(self.pending)>2*1024**2:raise OSError('界面状态报文过大')
                        while b'\n' in self.pending:
                            line,self.pending=self.pending.split(b'\n',1);message=json.loads(line)
                            if self.cancelled:message.update(phase='stopping',ready=False)
                            self.state.update(message)
            except (OSError,ValueError,RuntimeError) as e:
                self.lines.append('UI连接异常，已请求阻尼停止：%s\n'%e)
                self.stop()
        if not self.active:
            self.state.update(phase='stopped' if self.cancelled else ('finished' if self.process.returncode==0 else 'error'),ready=False,exit_code=self.process.returncode)
            self.cleanup()
        return self.state
    def take_logs(self):
        text=''.join(self.lines);self.lines=[];return text
    def cleanup(self):
        if self.active:raise RuntimeError('Cannot detach a running controller')
        if self.conn:self.conn.close();self.conn=None
        if self.master is not None:os.close(self.master);self.master=None
        if self.directory:self.directory.cleanup();self.directory=None
        if self.policy_log:self.policy_log.close();self.policy_log=None
        self.process=None

class Demo:
    """Explicit UI-only demonstration; cannot create a controller process."""
    def __init__(self,root):self.state=dict(phase='idle',mode='policy',learner={});self.active=False;self.lines=[];self.started=0
    def start(self,stage='online',learn=True,check=False):
        self.active=True;self.started=time.monotonic();self.state.update(phase='running',mode='policy',source='base',ready=True,stage=stage,generation=0,outcome=None)
        self.lines.append('演示模式：所有状态均为模拟，不打开硬件，不执行推理或训练。\n')
    def send(self,action):
        if action=='takeover':self.state.update(mode='human',source='human',generation=self.state.get('generation',0)+1)
        elif action=='resume':self.state.update(mode='policy',source='rlt',generation=self.state.get('generation',0)+1)
        elif action in ('success','failure'):self.active=False;self.state.update(phase='finished',ready=False,outcome=int(action=='success'))
    def stop(self):self.active=False;self.state.update(phase='stopped',ready=False)
    def poll(self):
        if self.active:self.state.update(rtt_ms=144.2,recorded_commands=int((time.monotonic()-self.started)*120),learner=dict(status='training',global_step=1240,transitions=768,human_transitions=156,actor_version=0,ready=False),joint_deg=[-3.9,.2,.2,7.5,13.8,-16.8],gripper_mm=35.)
        return self.state
    def take_logs(self):text=''.join(self.lines);self.lines=[];return text
    def cleanup(self):pass
