"""Keyboard-only subprocess: losing X11 cannot kill the CAN watchdog."""
import ctypes
import ctypes.util
import json
import os
import signal
import time


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    lib = ctypes.CDLL(ctypes.util.find_library('X11'))
    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    lib.XKeysymToKeycode.restype = ctypes.c_ubyte
    display = lib.XOpenDisplay(os.environ.get('DISPLAY', ':0').encode())
    if not display:
        raise RuntimeError('Global keyboard unavailable: cannot open X display')
    codes = [int(lib.XKeysymToKeycode(display, key)) for key in (0xFF1B, 0x20)]
    if not all(codes):
        raise RuntimeError('Global Escape/Space key mapping unavailable')
    while True:
        bits = ctypes.create_string_buffer(32)
        if not lib.XQueryKeymap(display, bits):
            raise RuntimeError('Global keyboard query failed')
        stop = any(bits.raw[key//8] & (1 << (key%8)) for key in codes)
        print(json.dumps(dict(t=time.monotonic(), stop=bool(stop))), flush=True)
        time.sleep(.015)


if __name__ == '__main__':
    main()
