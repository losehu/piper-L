"""Low-speed can0 positioning, only after full self-collision path validation.

No Disable, no reset, no gripper actuation. Firmware must support seamless
switching (>=1.8.5); otherwise alignment is refused before mode changes.
"""
import re,struct,threading,time
from pathlib import Path
import numpy as np
from rlt_piper.collision import Model
from rtc_piper.observations import NotReady
MEASURED_LIMIT_TOLERANCE=np.radians(.1)

def software_version(raw):
    # Official pyAgxArm get_firmware: eleven 8-byte frames; software is
    # bytes 60:68. Hardware and production date occupy separate fields.
    if len(raw)!=88 or not raw.startswith(b'H-V'):raise ValueError('主臂固件响应不完整或帧顺序异常')
    field=raw[60:68]
    match=re.fullmatch(rb'S-V(\d)\.(\d)-(\d)',field)
    if match is None:raise ValueError('主臂软件版本字段无效: '+repr(field))
    version=tuple(int(v) for v in match.groups())
    if version[0]!=1 or version<(1,8,5):raise ValueError('主臂软件版本不支持>=1.8.5无缝切换: '+field.decode('ascii'))
    return field.decode('ascii')

class Aligner:
    def __init__(self,leader,guard):
        self.leader,self.guard=leader,guard;self.model=Model(Path(__file__).with_name('leader_geometry.json'))
        self.model.cancelled=lambda:bool(self.guard.stop.value)
        self.thread=None;self.error=None;self.result=None;self.phase='idle';self.progress=0.
    def start(self,target):
        if self.thread and self.thread.is_alive():raise RuntimeError('Leader alignment already active')
        self.target=np.array(target,copy=True);self.error=None;self.result=None;self.phase='checking'
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def send(self,cid,data):
        if self.guard.stop.value:raise RuntimeError('Stop latched during leader alignment')
        if cid not in (0x4AF,0x472,0x151,0x155,0x156,0x157,0x470,0x191):raise RuntimeError('Unapproved leader packet')
        import can
        self.leader.bus.send(can.Message(arbitration_id=cid,data=bytes(data),is_extended_id=False),timeout=.02)
    def pause(self,seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if self.guard.stop.value:raise RuntimeError('Stop latched during leader alignment')
            if self.leader.error:raise RuntimeError(self.leader.error)
            time.sleep(min(.01,max(0,until-time.monotonic())))
    def firmware(self):
        after=time.monotonic();self.send(0x4AF,[1,0,0,0,0,0,0,0]);self.pause(.4)
        parts=self.leader.since(0x4AF,after);raw=b''.join(data for data,_ in parts)
        try:version=software_version(raw)
        except ValueError as e:raise RuntimeError(str(e)+'；不执行复位或卸力。反馈(hex): '+raw.hex()) from e
        print('主臂软件固件已确认：'+version+'（支持无缝切换）。',flush=True)
        return version
    def limits(self):
        low=[];high=[]
        for joint in range(1,7):
            after=time.monotonic();self.send(0x472,[joint,1,0,0,0,0,0,0]);until=after+.4;row=None
            while time.monotonic()<until:
                self.pause(.01)
                matches=[data for data,_ in self.leader.since(0x473,after) if data[0]==joint]
                if matches:row=matches[-1];break
            if row is None:raise RuntimeError('主臂未返回关节限位')
            maximum,minimum,speed=struct.unpack('>hhH',row[1:7]);low.append(np.radians(minimum/10));high.append(np.radians(maximum/10))
        self.model.lower=np.maximum(self.model.lower,low);self.model.upper=np.minimum(self.model.upper,high)
        if np.any(self.model.lower>=self.model.upper):raise RuntimeError('Invalid live leader joint limits')
    def joints(self,q):
        if self.model.collision(q):raise RuntimeError('主臂指令超限或碰撞，拒绝发送')
        values=np.rint(np.degrees(q)*1000).astype(np.int32)
        for i in range(3):self.send(0x155+i,struct.pack('>ii',int(values[2*i]),int(values[2*i+1])))
    def mode(self):self.send(0x151,[1,1,5,0,0,0,0,0])
    def ordinary(self,after=None):
        q,stamps=self.leader.feedback()
        self.leader.status()
        if after is not None and np.min(stamps)<=after:raise NotReady('Waiting for post-mode leader feedback')
        return q,stamps
    def measured_start(self,q):
        reason=self.model.collision(q,limit_tolerance=MEASURED_LIMIT_TOLERANCE)
        if reason:raise RuntimeError('主臂当前姿态未通过限位/防自碰撞检查（'+reason+'）；保持从臂停止，请先手动展开主臂再重新接管')
        bounded=np.clip(q,self.model.lower,self.model.upper)
        if not self.model.edge(q,bounded,limit_tolerance=MEASURED_LIMIT_TOLERANCE):raise RuntimeError('Leader measured-to-bounded start sweep is not collision-free')
        return bounded
    def run(self):
        try:
            self.firmware();self.limits();goal=self.target[:6]
            self.phase='countdown';self.pause(2.)
            for _ in range(3):
                # Native teaching commands may be silent at rest. Positioning
                # starts from measured joints, not cached teaching commands.
                start,_=self.leader.feedback();self.phase='planning';path=self.model.plan(self.measured_start(start[:6]),goal)
                fresh,_=self.leader.feedback()
                if np.max(np.abs(fresh[:6]-start[:6]))<=np.radians(.2):break
            else:raise RuntimeError('主臂在规划时持续移动，拒绝自动对齐')
            # Check the last refreshed start to the planned start as well.
            bounded_fresh=self.measured_start(fresh[:6])
            if not self.model.edge(fresh[:6],path[0],limit_tolerance=MEASURED_LIMIT_TOLERANCE):raise RuntimeError('Leader initial sweep is not collision-free')
            if np.max(np.abs(bounded_fresh-path[0]))>1e-9:path=[bounded_fresh]+path
            self.phase='mode_switch'
            self.guard.leader_active.value=1
            switched=time.monotonic();self.send(0x470,[0xFC,0,0,0,0,0,0,0]);self.mode();self.joints(path[0])
            until=switched+3
            while True:
                self.pause(.05);self.mode();self.joints(path[0])
                try:q,_=self.ordinary(switched);break
                except NotReady:
                    if time.monotonic()>=until:raise
            if np.max(np.abs(q[:6]-path[0]))>np.radians(1):raise RuntimeError('Leader moved during mode switch')
            self.phase='moving';total=len(path)-1
            for i,(a,b) in enumerate(zip(path,path[1:])):
                delta=float(np.max(np.abs(b-a)))
                duration=max(.5,1.875*delta/np.radians(5),np.sqrt(5.78*delta/np.radians(10)))
                began=time.monotonic();previous=a
                while True:
                    now=time.monotonic();s=min(1.,(now-began)/duration);blend=10*s**3-15*s**4+6*s**5;command=a+(b-a)*blend
                    measured,_=self.ordinary()
                    if self.model.collision(measured[:6],limit_tolerance=MEASURED_LIMIT_TOLERANCE) or not self.model.edge(measured[:6],command,limit_tolerance=MEASURED_LIMIT_TOLERANCE):raise RuntimeError('主臂实测/指令扫掠接近碰撞，停止')
                    if np.max(np.abs(measured[:6]-previous))>np.radians(2):raise RuntimeError('主臂跟踪误差过大，停止')
                    self.mode();self.joints(command);previous=command;self.progress=(i+s)/total
                    if s>=1:break
                    self.pause(.05)
            self.phase='settling';stable=None;until=time.monotonic()+5
            while time.monotonic()<until:
                measured,_=self.ordinary()
                if self.model.collision(measured[:6],limit_tolerance=MEASURED_LIMIT_TOLERANCE) or not self.model.edge(measured[:6],goal,limit_tolerance=MEASURED_LIMIT_TOLERANCE):raise RuntimeError('主臂到位保持的扫掠检查失败')
                self.mode();self.joints(goal)
                if np.max(np.abs(measured[:6]-goal))<=np.radians(.5):
                    stable=stable or time.monotonic()
                    if time.monotonic()-stable>=.3:break
                else:stable=None
                self.pause(.05)
            else:raise RuntimeError('Leader did not settle at follower pose')
            self.phase='restoring_drag';restored=time.monotonic();self.send(0x470,[0xFA,0,0,0,0,0,0,0]);self.send(0x191,[1,0,0,0,0,0,0,0])
            until=time.monotonic()+3;stable=None
            while time.monotonic()<until:
                self.pause(.02)
                try:q,stamps=self.leader.vector()
                except RuntimeError:continue
                if np.min(stamps)<=restored:continue
                if np.max(np.abs(q[:6]-goal))>np.radians(1):raise RuntimeError('主臂恢复拖动后姿态偏离，拒绝跟随')
                stable=stable or time.monotonic()
                if time.monotonic()-stable>=.3:self.result=(q.copy(),stamps.copy());self.phase='ready';return
            raise RuntimeError('No stable fresh native leader frames after drag restore')
        except BaseException as e:
            self.error=str(e);self.phase='failed';self.guard.stop.value=1
            print('主臂自动对齐失败，已请求停止：'+self.error,flush=True)
    def close(self):
        if self.thread:
            self.thread.join(2)
            if self.thread.is_alive():raise RuntimeError('Leader aligner has not stopped')
