"""F8/F9 outcome reader; independent of the unchanged damping stop supervisor."""
import ctypes,ctypes.util,os,threading,time

class Labels:
    def __init__(self,recorder,guard):self.recorder,self.guard=recorder,guard;self.stop=threading.Event();self.ready=threading.Event()
    def start(self):
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
        if not self.ready.wait(3) or self.recorder.error:raise RuntimeError('RLT outcome keyboard unavailable: '+str(self.recorder.error))
    def run(self):
        display=None
        try:
            lib=ctypes.CDLL(ctypes.util.find_library('X11'));lib.XOpenDisplay.argtypes=[ctypes.c_char_p];lib.XOpenDisplay.restype=ctypes.c_void_p
            lib.XQueryKeymap.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
            lib.XKeysymToKeycode.argtypes=[ctypes.c_void_p,ctypes.c_ulong];lib.XKeysymToKeycode.restype=ctypes.c_ubyte
            lib.XCloseDisplay.argtypes=[ctypes.c_void_p]
            display=lib.XOpenDisplay(os.environ.get('DISPLAY',':0').encode())
            if not display:raise RuntimeError('Cannot open X display')
            codes=[int(lib.XKeysymToKeycode(display,k)) for k in (0xFFC5,0xFFC6)]
            if not all(codes):raise RuntimeError('F8/F9 key mapping unavailable')
            last=None
            while not self.stop.is_set():
                bits=ctypes.create_string_buffer(32)
                if not lib.XQueryKeymap(display,bits):raise RuntimeError('Keyboard query failed')
                pressed=tuple(bool(bits.raw[k//8]&(1<<(k%8))) for k in codes)
                if last is not None:
                    for i,on in enumerate(pressed):
                        if on and not last[i]:self.recorder.outcome(1-i,self.guard)
                last=pressed;self.ready.set();self.stop.wait(.015)
        except BaseException as e:self.recorder.error=repr(e);self.guard.stop.value=1;self.ready.set()
        finally:
            if display:lib.XCloseDisplay(display)
    def close(self):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join(1)
