"""Collector-compatible firmware home, then ordinary low-speed can0 MOVEJ.

The uncommissioned box model is retained as an offline diagnostic only.
All transmitted joint targets use the live firmware limits. No gripper TX.
"""
import re,struct,threading,time
import numpy as np
from rtc_piper.observations import NotReady

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
        self.leader,self.guard=leader,guard;self.lower=self.upper=None;self.homed=False
        self.thread=None;self.error=None;self.result=None;self.phase='idle';self.progress=0.
    def start(self,target):
        if self.thread and self.thread.is_alive():raise RuntimeError('Leader alignment already active')
        self.target=np.array(target,copy=True);self.error=None;self.result=None;self.phase='checking'
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    def send(self,cid,data):
        if self.guard.stop.value:raise RuntimeError('Stop latched during leader alignment')
        approved=cid in (0x4AF,0x472,0x151,0x155,0x156,0x157,0x470,0x191)
        if cid==0x471:approved=bytes(data)==bytes([7,2,0,0,0,0,0,0])
        if cid==0x150:approved=bytes(data)==bytes([2,0,0,0,0,0,0,0])
        if not approved:raise RuntimeError('Unapproved leader packet')
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
        low=[];high=[];speeds=[]
        for joint in range(1,7):
            after=time.monotonic();self.send(0x472,[joint,1,0,0,0,0,0,0]);until=after+.4;row=None
            while time.monotonic()<until:
                self.pause(.01)
                matches=[data for data,_ in self.leader.since(0x473,after) if data[0]==joint]
                if matches:row=matches[-1];break
            if row is None:raise RuntimeError('主臂未返回关节限位')
            maximum,minimum,speed=struct.unpack('>hhH',row[1:7]);low.append(minimum*100);high.append(maximum*100);speeds.append(speed/1000)
        self.lower_mdeg=np.asarray(low);self.upper_mdeg=np.asarray(high)
        self.lower=np.radians(self.lower_mdeg/1000);self.upper=np.radians(self.upper_mdeg/1000)
        if np.any(self.lower>=self.upper) or np.any(np.abs(np.r_[self.lower,self.upper])>2*np.pi):raise RuntimeError('Invalid live leader joint limits')
        self.max_speed=np.asarray(speeds)
        if np.any(self.max_speed<=0):raise RuntimeError('Invalid live leader speed limits')
        self.move_percent=min(100,max(1,int(np.floor(np.radians(5)/max(self.max_speed)*100))))
        self.path_speed=np.minimum(np.radians(5),self.max_speed*self.move_percent/100*.9)
    def bounded(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise RuntimeError('Invalid leader joint target')
        if self.lower is None:raise RuntimeError('Leader firmware limits not loaded')
        return np.clip(q,self.lower,self.upper)
    def joints(self,q):
        values=np.clip(np.rint(np.degrees(self.bounded(q))*1000),self.lower_mdeg,self.upper_mdeg).astype(np.int32)
        for i in range(3):self.send(0x155+i,struct.pack('>ii',int(values[2*i]),int(values[2*i+1])))
    def mode(self):self.send(0x151,[1,1,self.move_percent,0,0,0,0,0])
    def ordinary(self,after=None):
        q,stamps=self.leader.feedback()
        self.leader.status()
        if after is not None and np.min(stamps)<=after:raise NotReady('Waiting for post-mode leader feedback')
        return q,stamps
    def native_home(self):
        """Same 0x191 bytes and quiet-at-zero completion as piper_teleop."""
        self.phase='native_home';self.guard.leader_active.value=1
        print('主臂按数采流程厂家回零：速度由固件控制；从臂保持，Space/Esc阻尼停止。',flush=True)
        try:stopped=self.leader.get([0x2A1],.25)[0][0][1]==1
        except NotReady:stopped=False
        if stopped:
            print('恢复主臂固件急停，请保持支撑。',flush=True)
            self.send(0x150,[2,0,0,0,0,0,0,0]);self.pause(.3)
        self.send(0x470,[0xFA,0,0,0,0,0,0,0]);self.pause(.15)
        self.send(0x191,[1,0,0,0,0,0,0,0])
        self.leader.invalidate_native();self.leader.reset_link();sent=time.monotonic()
        # mode=2 retains native reports, on isolated can0 only; no CAN bridge.
        self.send(0x191,[1,0,1,0,0,0,0,0])
        last_source=None;last_raw=None;zero_since=None;last_zero_wall=None;zero_result=None;reports=0
        while time.monotonic()-sent<30:
            self.pause(.01);self.leader.check_link();now=time.monotonic()
            try:q,stamps=self.leader.vector(max_age=float('inf'))
            except NotReady:continue
            source=float(np.min(stamps))
            if source<=sent:continue
            if source==last_source:
                if zero_result is not None and reports>=2 and now-last_zero_wall>=.6:
                    self.send(0x191,[1,0,0,0,0,0,0,0]);self.homed=True;return
                continue
            reports+=1;last_source=source;error=float(np.max(np.abs(np.degrees(q[:6]))))
            stable=error<=1 and (last_raw is None or np.max(np.abs(np.degrees(q[:6]-last_raw)))<=.1)
            zero_since=(zero_since or source) if stable else None;last_raw=q[:6].copy()
            if error<=1:zero_result=q.copy();last_zero_wall=now
            else:zero_result=None;last_zero_wall=None
            if zero_since is not None and source-zero_since>=.3:
                self.send(0x191,[1,0,0,0,0,0,0,0]);self.homed=True;return
        raise RuntimeError('主臂厂家回零30秒未确认完成')
    def run(self):
        try:
            self.firmware();self.limits();goal=self.bounded(self.target[:6])
            self.phase='countdown';self.pause(2.)
            if not self.homed:self.native_home()
            start,_=self.leader.feedback();path=[self.bounded(start[:6]),goal]
            self.phase='mode_switch'
            self.guard.leader_active.value=1
            switched=time.monotonic();self.send(0x470,[0xFC,0,0,0,0,0,0,0]);self.mode();self.joints(path[0])
            until=switched+3;enable_at=float('-inf')
            while True:
                self.pause(.05);self.mode();self.joints(path[0])
                if time.monotonic()-enable_at>=.2:
                    self.send(0x471,[7,2,0,0,0,0,0,0]);enable_at=time.monotonic()
                try:q,_=self.ordinary(switched);break
                except NotReady:
                    if time.monotonic()>=until:raise
            self.phase='moving';total=len(path)-1
            for i,(a,b) in enumerate(zip(path,path[1:])):
                delta=float(np.max(np.abs(b-a)))
                duration=max(.5,1.875*float(np.max(np.abs(b-a)/self.path_speed)),np.sqrt(5.78*delta/np.radians(10)))
                began=time.monotonic();previous=a
                while True:
                    now=time.monotonic();s=min(1.,(now-began)/duration);blend=10*s**3-15*s**4+6*s**5;command=a+(b-a)*blend
                    measured,_=self.ordinary()
                    self.mode();self.joints(command);previous=command;self.progress=(i+s)/total
                    if s>=1:break
                    self.pause(.05)
            self.phase='settling';stable=None;until=time.monotonic()+5
            while time.monotonic()<until:
                measured,_=self.ordinary()
                self.mode();self.joints(goal)
                if np.max(np.abs(measured[:6]-goal))<=np.radians(.5):
                    stable=stable or time.monotonic()
                    if time.monotonic()-stable>=.3:break
                else:stable=None
                self.pause(.05)
            else:raise RuntimeError('Leader did not settle at follower pose')
            self.phase='restoring_drag';self.leader.invalidate_native();self.leader.reset_link();restored=time.monotonic();self.send(0x470,[0xFA,0,0,0,0,0,0,0]);self.send(0x191,[1,0,0,0,0,0,0,0])
            until=time.monotonic()+3;stable=None
            while time.monotonic()<until:
                self.pause(.02);self.leader.check_link()
                try:q,stamps=self.leader.vector(max_age=float('inf'))
                except NotReady:continue
                if np.min(stamps)<=restored:continue
                if np.max(np.abs(q[:6]-goal))>np.radians(1):raise RuntimeError('主臂恢复拖动后姿态偏离，拒绝跟随')
                stable=stable or time.monotonic()
                if time.monotonic()-stable>=.3:self.result=(q.copy(),stamps.copy());self.phase='ready';return
            raise RuntimeError('No stable fresh native leader frames after drag restore')
        except BaseException as e:
            self.error=self.phase+': '+str(e);self.phase='failed';self.guard.stop.value=1
            print('主臂自动对齐失败，已请求停止：'+self.error,flush=True)
    def close(self):
        if self.thread:
            self.thread.join(2)
            if self.thread.is_alive():raise RuntimeError('Leader aligner has not stopped')
