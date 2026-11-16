"""Additional independent can0 damping supervisor for powered leader alignment.

The original can1 Guard and keyboard process remain unchanged.
"""
import multiprocessing as mp,os,time
from rtc_piper.safety import Guard as FollowerGuard,damped_stop

def leader_supervisor(parent,active,stop,closed,heartbeat):
    while not closed.value and not stop.value:
        if os.getppid()!=parent or (active.value and time.monotonic()-heartbeat.value>.35):stop.value=1;break
        time.sleep(.005)
    if active.value:
        stop.value=1
        print('主臂阻尼停止:',damped_stop('can0'),flush=True)

class Guard(FollowerGuard):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);ctx=mp.get_context('spawn')
        self.leader_active=ctx.RawValue('b',0)
        self.leader_process=ctx.Process(target=leader_supervisor,args=(os.getpid(),self.leader_active,self.stop,self.closed,self.heartbeat),daemon=False)
    def start(self):
        super().start();self.leader_process.start()
    def check(self):
        super().check()
        if self.leader_active.value and not self.leader_process.is_alive():self.stop.value=1;raise RuntimeError('Leader damping supervisor unavailable')
    def close(self):
        try:super().close()
        finally:
            if self.leader_process.pid is not None:
                self.leader_process.join(2)
                if self.leader_process.is_alive():raise RuntimeError('Leader damping supervisor still stopping')
