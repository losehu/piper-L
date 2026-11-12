"""Lock-free independent keyboard/heartbeat supervisor. Stop-only CAN transport."""
import contextlib
import json
import multiprocessing as mp
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import termios
import time
import tty


def damped_stop(channel):
    """Only 0x150/01; never Disable, Reset, motion mode or a gripper command."""
    packet = struct.pack('=IB3x8s', 0x150, 8, bytes([1,0,0,0,0,0,0,0]))
    errors, sent = [], 0
    for _ in range(10):
        try:
            with socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW) as bus:
                bus.settimeout(.02)
                bus.bind((channel,))
                if bus.send(packet) != 16:
                    raise OSError('Incomplete stop packet')
                sent += 1
        except Exception as error:
            errors.append(str(error))
        time.sleep(.04)
    return dict(sent=sent, errors=errors, hardware_acknowledged=False)


def supervisor(parent, flags, heartbeat, policy_time, startup_deadline, channel):
    stop, armed, running, ready, closed, terminal_keys = flags
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_:setattr(stop, 'value', 1))
    child = None
    terminal = None
    reason, last_key, pending = '', None, b''
    try:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'rtc_piper.keys'],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        begun = time.monotonic()
        try:
            terminal = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            pass
        while not closed.value:
            now = time.monotonic()
            if select.select([child.stdout], [], [], 0)[0]:
                pending += os.read(child.stdout.fileno(), 8192)
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    item = json.loads(line)
                    if not -.01 <= time.monotonic()-item['t'] <= .5:
                        raise RuntimeError('Global keyboard heartbeat stale')
                    last_key = item['t']
                    if item['stop']:
                        reason = 'Global Escape/Space'
                        stop.value = 1
            if terminal_keys.value and terminal is not None and select.select([terminal],[],[],0)[0]:
                data = os.read(terminal, 256)
                if any(key in data for key in (b' ', b'\x1b', b'\x03')):
                    reason = 'Terminal Escape/Space/Ctrl+C'
                    stop.value = 1
            if child.poll() is not None:
                raise RuntimeError('Global keyboard process exited')
            if (last_key is None and now-begun > 5) or (last_key is not None and now-last_key > .5):
                raise RuntimeError('Global keyboard unavailable')
            if last_key is not None and not stop.value:
                ready.value = 1
            if os.getppid() != parent:
                reason = 'Controller process exited'
                stop.value = 1
            if armed.value:
                if now-heartbeat.value > .35:
                    reason = 'Control heartbeat timeout'
                    stop.value = 1
                if running.value and now-policy_time.value > .8:
                    reason = 'Policy heartbeat timeout'
                    stop.value = 1
                if not running.value and now > startup_deadline.value:
                    reason = 'Startup deadline exceeded'
                    stop.value = 1
            if stop.value:
                break
            time.sleep(.005)
    except BaseException as error:
        reason = str(error)
        stop.value = 1
    finally:
        ready.value = 0
        # Latch precedes every I/O and the independently encoded damping path.
        if armed.value:
            stop.value = 1
            result = damped_stop(channel)
            print('STOP:', reason or 'Requested/closed', result, flush=True)
        if child:
            child.terminate()
            try: child.wait(timeout=.5)
            except subprocess.TimeoutExpired: child.kill()
        if terminal is not None:
            os.close(terminal)


class Guard:
    def __init__(self, channel):
        if channel != 'can1':
            raise ValueError('This deployment is bound to the verified follower can1')
        ctx = mp.get_context('spawn')
        self.flags = [ctx.RawValue('b', 0) for _ in range(6)]
        self.stop, self.armed, self.running, self.ready, self.closed, self.terminal_keys = self.flags
        self.heartbeat = ctx.RawValue('d', time.monotonic())
        self.policy_time = ctx.RawValue('d', time.monotonic())
        self.startup_deadline = ctx.RawValue('d', float('inf'))
        self.process = ctx.Process(target=supervisor,
            args=(os.getpid(), self.flags, self.heartbeat, self.policy_time,
                  self.startup_deadline, channel), daemon=False)

    def start(self):
        self.process.start()
        until = time.monotonic()+6
        while not self.ready.value:
            if self.stop.value or not self.process.is_alive() or time.monotonic()>until:
                raise RuntimeError('Global emergency-stop keyboard did not become ready')
            time.sleep(.01)

    def check(self):
        if self.stop.value or not self.ready.value or not self.process.is_alive():
            self.stop.value = 1
            raise RuntimeError('Emergency stop latched or independent guard unavailable')
        self.heartbeat.value = time.monotonic()

    def arm(self, startup_seconds=20):
        self.check()
        self.startup_deadline.value = time.monotonic()+startup_seconds
        self.armed.value = 1
        self.check()

    def close(self):
        self.stop.value = 1
        self.closed.value = 1
        if self.process.pid is not None:
            self.process.join(2)
            if self.process.is_alive():
                raise RuntimeError('Damping supervisor still stopping; do not restart')


@contextlib.contextmanager
def terminal_mode(guard):
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        raise RuntimeError('Run from an interactive terminal')
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        guard.terminal_keys.value = 1
        yield
    finally:
        guard.terminal_keys.value = 0
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
