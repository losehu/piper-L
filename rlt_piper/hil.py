"""Native leader input, read-only link probes and generation-based handover."""
import fcntl,json,math,struct,threading,time
from collections import defaultdict,deque
from rtc_piper.observations import NotReady
from pathlib import Path
import numpy as np

class Leader:
    def __init__(self,config=Path('/home/arm/piper_teleop/config.json')):
        import can
        import subprocess
        self.cfg=json.loads(Path(config).read_text())
        if self.cfg['leader']!='can0' or self.cfg['follower']!='can1':raise RuntimeError('Unverified leader/follower channels')
        props=subprocess.check_output(['udevadm','info','-q','property','-p','/sys/class/net/can0'],text=True)
        if 'ID_SERIAL='+self.cfg['adapter_serials']['can0'] not in props.splitlines():raise RuntimeError('Leader USB-CAN identity mismatch')
        link=json.loads(subprocess.check_output(['ip','-j','-details','link','show','can0']))[0]
        if 'UP' not in link['flags'] or link['linkinfo']['info_data']['bittiming']['bitrate']!=1000000:raise RuntimeError('Leader must already be UP at 1Mbps')
        self.file=(Path(self.cfg['lock_root'])/'.can0.lock').open('a')
        try:fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.file.close();raise
        try:self.bus=can.Bus(channel='can0',interface='socketcan',receive_own_messages=False)
        except BaseException:self.file.close();raise
        self.lock=threading.Lock();self.frames={};self.history=defaultdict(lambda:deque(maxlen=128));self.error=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        try:
            while not self.stop.is_set():
                m=self.bus.recv(.05)
                if m is None:continue
                if m.is_error_frame:raise RuntimeError('Leader CAN error')
                if m.is_extended_id or m.is_remote_frame or len(m.data)!=8:continue
                stamp=time.monotonic()-max(0.,time.time()-m.timestamp)
                with self.lock:
                    self.frames[m.arbitration_id]=(bytes(m.data),stamp);self.history[m.arbitration_id].append((bytes(m.data),stamp))
        except BaseException as e:self.error=str(e)
    def since(self,cid,after):
        with self.lock:return [(d,t) for d,t in self.history[cid] if t>after]
    def get(self,ids,max_age=.15):
        if self.error:raise RuntimeError(self.error)
        with self.lock:frames=[self.frames.get(k) for k in ids]
        now=time.monotonic()
        bad=[('0x%03X'%cid)+(' 缺失' if frame is None else ' 超时 %.0fms'%((now-frame[1])*1000)) for cid,frame in zip(ids,frames) if frame is None or not -.01<=now-frame[1]<=max_age]
        if bad:raise NotReady('主臂报文缺失/超时：'+', '.join(bad))
        return frames
    def vector(self,max_age=.15):
        frames=self.get((0x155,0x156,0x157,0x159),max_age);q=np.array([v for data,_ in frames[:3] for v in struct.unpack('>ii',data)])*math.pi/180000
        if max(t for _,t in frames)-min(t for _,t in frames)>.05:raise NotReady('主臂示教帧时间差超过50ms')
        data=frames[3][0]
        if (data[6]>>2)&1:raise RuntimeError('Leader gripper must use travel mode')
        width=struct.unpack('>i',data[:4])[0]/1e6-self.cfg['leader_gripper_closed_raw']/1e6
        if not math.isfinite(width) or not -.02<=width<=self.cfg['gripper_max_m']+.02:raise RuntimeError('Leader gripper mapping invalid')
        width=np.clip(width*float(self.cfg.get('gripper_scale',1)),0,self.cfg['gripper_max_m'])
        result=np.r_[q,width],np.array([f[1] for f in frames]);self.held_native=(result[0].copy(),result[1].copy())
        return result
    def invalidate_native(self):
        with self.lock:
            for cid in (0x155,0x156,0x157,0x159):self.frames.pop(cid,None)
        self.held_native=None;self.partial_since=None
    def reset_link(self):
        self.link_started=time.monotonic();self.link_response=self.link_started;self.link_query=float('-inf')
    def check_link(self):
        """Collector's 5Hz firmware ping; no enable or movement frames."""
        import can
        if self.error:raise RuntimeError(self.error)
        if not hasattr(self,'link_started'):self.reset_link()
        now=time.monotonic()
        with self.lock:
            reply=self.frames.get(0x4AF);status=self.frames.get(0x2A1)
            drivers=[self.frames.get(cid) for cid in range(0x261,0x267)]
        if status and now-status[1]<=.25 and (status[0][1] or int.from_bytes(status[0][6:8],'big')):raise RuntimeError('主臂固件保护/故障反馈')
        if any(row and now-row[1]<=.25 and row[0][5]&0xBF for row in drivers):raise RuntimeError('主臂驱动故障反馈')
        if reply and reply[1]>self.link_started:self.link_response=max(self.link_response,reply[1])
        if now-self.link_query>=.2:
            self.bus.send(can.Message(arbitration_id=0x4AF,data=bytes([1,0,0,0,0,0,0,0]),is_extended_id=False),timeout=.02);self.link_query=now
        if now-self.link_response>.5:raise RuntimeError('can0 在线查询超过500ms无回复')
    def following(self):
        self.check_link()
        try:
            result=self.vector();self.partial_since=None;return result
        except NotReady:
            now=time.monotonic()
            with self.lock:recent=any(cid in self.frames and now-self.frames[cid][1]<.15 for cid in (0x155,0x156,0x157,0x159))
            if recent:
                self.partial_since=getattr(self,'partial_since',None) or now
                if now-self.partial_since>.05:raise RuntimeError('主臂部分示教帧持续缺失')
            else:self.partial_since=None
            held=getattr(self,'held_native',None)
            if held is None:raise
            # Silent-at-rest is normal. Preserve original sample timestamps;
            # only the independent query establishes current link liveness.
            return held[0].copy(),held[1].copy()
    def feedback(self):
        frames=self.get((0x2A5,0x2A6,0x2A7))
        q=np.array([v for data,_ in frames for v in struct.unpack('>ii',data)])*math.pi/180000
        return np.r_[q,0.],np.array([f[1] for f in frames])
    def status(self):
        frames=self.get([0x2A1]+list(range(0x261,0x267)),.25);data=frames[0][0]
        if data[1]!=0 or int.from_bytes(data[6:8],'big') or any(d[5]&0xBF for d,_ in frames[1:]):raise RuntimeError('主臂保护/驱动故障')
        if data[0]!=1 or data[2]!=1 or not all(d[5]&0x40 for d,_ in frames[1:]):raise NotReady('Waiting for enabled leader CAN/MOVEJ feedback')
    def close(self):
        self.stop.set();self.thread.join(1);self.bus.shutdown();self.file.close()

class Handover:
    def __init__(self,aligner=None):
        self.lock=threading.RLock();self.requested=False;self.generation=0;self.mode='policy';self.after=float('-inf')
        self.anchor=None;self.follow=None;self.aligner=aligner;self.samples=deque(maxlen=50);self.stable_since=None;self.align_started=False
    def toggle(self):
        with self.lock:self.requested=True
    def switch(self,measured,leader,now,previous_command=None):
        """Freeze the follower first; never sample/move the master on this path."""
        self.generation+=1;self.after=now;self.requested=False
        if self.mode=='human':self.mode='policy';self.anchor=None
        elif self.mode=='aligning':raise RuntimeError('主臂对齐中再次按I，已请求停止；请重新开始本轮')
        else:
            self.follow=np.array(measured,copy=True);self.mode='aligning';self.samples.clear();self.stable_since=None;self.align_started=False
            previous=self.follow if previous_command is None else np.asarray(previous_command)
            if previous.shape!=self.follow.shape or not np.isfinite(previous).all():raise RuntimeError('Invalid pre-takeover command')
            # Until the servo has stopped, residual travel may remain between
            # measured position and the last command. Never execute that old
            # command again; use it only to bound the stopping observation.
            self.stop_lower=np.minimum(self.follow[:6],previous[:6])-np.radians(1)
            self.stop_upper=np.maximum(self.follow[:6],previous[:6])+np.radians(1)
        return self.mode
    def poll_alignment(self,measured,now):
        if self.aligner is None:raise RuntimeError('Leader auto alignment backend unavailable')
        if not self.align_started:
            if np.any(measured[:6]<self.stop_lower) or np.any(measured[:6]>self.stop_upper):raise RuntimeError('从臂停稳过程超出接管前指令范围，未启动主臂对齐')
            self.samples.append((now,np.array(measured,copy=True)))
            recent=[q for t,q in self.samples if now-t<=.22]
            stationary=(self.samples and now-self.samples[0][0]>=.2 and len(recent)>2
                        and np.max(np.ptp(np.array(recent)[:,:6],axis=0))<=np.radians(.1)
                        and np.max(np.abs(measured[:6]-self.follow[:6]))<=np.radians(.5)
                        and abs(measured[6]-self.follow[6])<=.003)
            self.stable_since=(self.stable_since or now) if stationary else None
            if self.stable_since is not None and now-self.stable_since>=.3:
                self.aligner.start(self.follow);self.align_started=True
                print('从臂实测已停稳并保持：开始主臂对齐检查。',flush=True)
            elif now-self.after>3:raise RuntimeError('从臂未稳定停止，未启动主臂对齐')
            return False
        if np.max(np.abs(measured[:6]-self.follow[:6]))>np.radians(1):raise RuntimeError('从臂保持姿态偏移，停止主臂对齐')
        if self.aligner.error:raise RuntimeError(self.aligner.error)
        if self.aligner.result is None:return False
        q,stamps=self.aligner.result
        self.anchor=np.array(q,copy=True);self.leader_times=np.array(stamps,copy=True)
        self.mode='human';self.generation+=1;self.after=now
        return True
    def target(self,leader):
        q,self.leader_times=getattr(leader,'following',leader.vector)();target=np.array(q,copy=True)
        # Six joints are now physically aligned. The passive teaching handle is
        # NOT motor-driven: only gripper input retains a clutch offset.
        target[6]=self.follow[6]+q[6]-self.anchor[6]
        return target,q,self.leader_times
    def accepts(self,generation,observed_at):
        return generation==self.generation and observed_at>self.after and not self.requested
