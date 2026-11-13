"""Read-only native leader input and generation-based handover. No CAN TX path."""
import fcntl,json,math,struct,threading,time
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
        self.lock=threading.Lock();self.frames={};self.error=None;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        try:
            while not self.stop.is_set():
                m=self.bus.recv(.05)
                if m is None:continue
                if m.is_error_frame:raise RuntimeError('Leader CAN error')
                if m.is_extended_id or m.is_remote_frame or len(m.data)!=8:continue
                if m.arbitration_id not in (0x155,0x156,0x157,0x159):continue
                stamp=time.monotonic()-max(0.,time.time()-m.timestamp)
                with self.lock:self.frames[m.arbitration_id]=(bytes(m.data),stamp)
        except BaseException as e:self.error=str(e)
    def vector(self):
        if self.error:raise RuntimeError(self.error)
        with self.lock:frames=[self.frames.get(k) for k in (0x155,0x156,0x157,0x159)]
        if any(f is None or not -.01<=time.monotonic()-f[1]<=.15 for f in frames):raise RuntimeError('主臂控制帧缺失/超时；请先在数采程序配置好主臂拖动模式')
        q=np.array([v for data,_ in frames[:3] for v in struct.unpack('>ii',data)])*math.pi/180000
        data=frames[3][0]
        if (data[6]>>2)&1:raise RuntimeError('Leader gripper must use travel mode')
        width=struct.unpack('>i',data[:4])[0]/1e6-self.cfg['leader_gripper_closed_raw']/1e6
        if not math.isfinite(width) or not -.02<=width<=self.cfg['gripper_max_m']+.02:raise RuntimeError('Leader gripper mapping invalid')
        width=np.clip(width*float(self.cfg.get('gripper_scale',1)),0,self.cfg['gripper_max_m'])
        return np.r_[q,width],np.array([f[1] for f in frames])
    def close(self):
        self.stop.set();self.thread.join(1);self.bus.shutdown();self.file.close()

class Handover:
    def __init__(self):
        self.lock=threading.RLock();self.requested=False;self.generation=0;self.mode='policy';self.after=float('-inf')
        self.anchor=None;self.follow=None
    def toggle(self):
        with self.lock:self.requested=True
    def switch(self,previous,leader,now):
        """Called by sole TX owner; initial human target equals last sent command."""
        self.generation+=1;self.after=now;self.requested=False
        if self.mode=='human':self.mode='policy';self.anchor=None
        else:
            anchor,self.leader_times=leader.vector();self.anchor=np.array(anchor,copy=True);self.follow=np.array(previous,copy=True);self.mode='human'
        return self.mode
    def target(self,leader):
        q,self.leader_times=leader.vector()
        return self.follow+(q-self.anchor),q,self.leader_times
    def accepts(self,generation,observed_at):
        return generation==self.generation and observed_at>self.after and not self.requested
