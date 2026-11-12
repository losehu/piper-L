"""Official pyAgxArm Piper-L/V189 bridge; each encoded TX passes the stop gate."""
import contextlib
from collections import deque
import fcntl
import json
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import numpy as np
from rtc_piper.observations import NotReady


class JointCommunicationFault(RuntimeError):
    """A reported communication fault, fatal except while awaiting reset completion."""
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


@contextlib.contextmanager
def follower_lock(config):
    path=Path(config['lock_file']).expanduser()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Follower is owned by another controller')
        yield


def check_interface(config):
    channel = config['channel']
    if channel != 'can1': raise ValueError('Verified follower is can1')
    props = subprocess.check_output(['udevadm','info','-q','property','-p',
                                    '/sys/class/net/'+channel],text=True)
    if 'ID_SERIAL='+config['adapter_serial'] not in props.splitlines():
        raise RuntimeError('Follower USB-CAN identity mismatch')
    info = json.loads(subprocess.check_output(['ip','-j','-details','link','show',channel]))[0]
    if ('UP' not in info['flags']
            or info['linkinfo']['info_data']['bittiming']['bitrate'] != 1000000):
        raise RuntimeError('Follower must already be UP at 1 Mbps')


def apply_hardware_limits(config, readback):
    rows=sorted(readback['limits'],key=lambda row:row['joint'])
    if [r['joint'] for r in rows]!=list(range(1,7)):raise ValueError('Incomplete hardware limits')
    lower=np.radians([r['lower_deg'] for r in rows]);upper=np.radians([r['upper_deg'] for r in rows])
    if (not np.isfinite(np.r_[lower,upper]).all() or np.any(lower>=upper)
            or np.any(np.abs(np.r_[lower,upper])>2*np.pi)):
        raise ValueError('Malformed firmware limit response')
    result=dict(config,hardware_limits_cache=readback,
                policy_command_lower_rad=lower.tolist(),policy_command_upper_rad=upper.tolist())
    return result


def read_hardware_limits(config):
    """Six read-only 0x472/type1 queries. No motion, enable, reset or mode TX."""
    rows=[]
    with socket.socket(socket.PF_CAN,socket.SOCK_RAW,socket.CAN_RAW) as bus:
        bus.settimeout(.02)
        bus.setsockopt(socket.SOL_CAN_RAW,socket.CAN_RAW_FILTER,struct.pack('=II',0x473,0x7ff))
        bus.bind((config['channel'],))
        for joint in range(1,7):
            query=struct.pack('=IB3x8s',0x472,8,bytes([joint,1,0,0,0,0,0,0]))
            if bus.send(query)!=16:raise RuntimeError('Incomplete hardware limit query')
            until=time.monotonic()+.5
            while time.monotonic()<until:
                try:packet=bus.recv(16)
                except socket.timeout:continue
                cid,size,data=struct.unpack('=IB3x8s',packet)
                if cid==0x473 and size==8 and data[0]==joint:
                    high,low,speed=struct.unpack('>hhH',data[1:7])
                    rows.append(dict(joint=joint,lower_deg=low/10,upper_deg=high/10,
                                     max_speed_rad_s=speed/1000,raw=data.hex()))
                    break
            else:raise RuntimeError('No fresh firmware limit response for J%d'%joint)
    result=dict(time=time.time(),limits=rows,query_frames_sent=6,
                source='live0x472/type1 -> 0x473, no controller parameters changed')
    apply_hardware_limits(config,result)  # Validate before returning any limits.
    return result


def gated_send(bus, guard, message, timeout=None):
    """The SDK still encodes all control frames. Never mask ENOBUFS or block TX."""
    guard.check()
    if not guard.armed.value:
        raise RuntimeError('Read-only stage: CAN TX is forbidden')
    if message.arbitration_id == 0x471 and len(message.data)>1 and message.data[1] != 2:
        raise RuntimeError('Disable commands are not part of this deployment')
    bus.send(message, timeout=.02)


class SdkArm:
    def __init__(self, config, bank, guard):
        from pyAgxArm import create_agx_arm_config, AgxArmFactory
        # The same live firmware limits are used in all motion phases.
        limits = {'joint%d'%(i+1):[config['policy_command_lower_rad'][i],config['policy_command_upper_rad'][i]]
                  for i in range(6)}
        sdk_config = create_agx_arm_config(
            robot='piper_l',firmeware_version='v189',comm='can',
            channel=config['channel'],interface='socketcan',bitrate=1000000,
            auto_connect=False,enable_check_can=False,joint_limits=limits)
        self.arm = AgxArmFactory.create_arm(sdk_config)
        self.arm.set_auto_set_motion_mode_enabled(False)
        # Match the verified collector's 0x151 [1,1,100,0xAD,0,0,0,0].
        # Set the local cache, so no extra setup packet is sent before RUN.
        self.arm._msg_mode.move_spd_rate_ctrl = 100
        self.bank, self.guard = bank, guard
        self.closed, self.errors = threading.Event(), []
        self._last_mode = float('-inf')
        self._last_joint_send = None
        self._control_profile = 'prepare'
        self._policy_lower=np.asarray(config['policy_command_lower_rad'])
        self._policy_upper=np.asarray(config['policy_command_upper_rad'])
        self._can_trace = deque(maxlen=8192)
        self._last_packet = {}
        self.thread = None
        self.arm.get_context().set_tx_repeat_min_interval(0x151,0)
        self.arm.get_context().register_parser_packet_fun(lambda message:self._trace('rx',message))
        comm = self.arm._ctx.init_comm()
        def send(message,timeout=None):
            gated_send(comm.send_bus,guard,message,timeout)
            self._trace('tx_queued',message)
        comm.send = send
        self.arm.connect()
        self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.thread = threading.Thread(target=self._read,daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.closed.is_set():
                parser = self.arm._parser  # Pinned SDK cache: per-CAN-pair timestamps.
                offset = time.time()-time.monotonic()
                for key,attr,fields in [('j12','joint_12',['joint_1','joint_2']),
                                        ('j34','joint_34',['joint_3','joint_4']),
                                        ('j56','joint_56',['joint_5','joint_6'])]:
                    packet = getattr(parser,attr,None)
                    if packet is not None:
                        before = packet.timestamp
                        values = [getattr(packet.msg,name) for name in fields]
                        if before == packet.timestamp and before > self._last_packet.get(key,0):
                            self.bank.state(key,before-offset,values)
                            self._last_packet[key] = before
                packet = self.gripper.get_gripper_status()
                if packet is not None:
                    if packet.msg.mode != 'width':
                        raise RuntimeError('Gripper must use width mode, not angle mode')
                    before = packet.timestamp
                    value = float(packet.msg.value)
                    if before == packet.timestamp and before > self._last_packet.get('grip',0):
                        self.bank.state('grip',before-offset,[value])
                        self._last_packet['grip'] = before
                self.closed.wait(.003)
        except Exception as error:
            self.errors.append(str(error))

    def status(self, allow_estop=False, require_enabled=False, after=None):
        if self.errors: raise RuntimeError('SDK feedback failure: '+str(self.errors))
        if self.arm.has_comm_error(): raise RuntimeError(str(self.arm.get_comm_error()))
        status = self.arm.get_arm_status()
        drivers = [self.arm.get_driver_states(i) for i in range(1,7)]
        grip = self.gripper.get_gripper_status()
        packets = [status,*drivers,grip]
        now = time.time()
        if any(p is None or not -.01 <= now-p.timestamp <= .25 for p in packets):
            raise NotReady('Waiting for fresh arm/driver/gripper status')
        if after is not None and any(p.timestamp <= after for p in packets):
            raise NotReady('Waiting for status newer than the startup command')
        code, error = int(status.msg.arm_status), int(status.msg.err_code)
        enabled = []
        for packet in drivers:
            flags = packet.msg.foc_status
            if any(value for key,value in vars(flags).items() if key!='driver_enable_status'):
                raise RuntimeError('Joint driver fault: '+str(flags))
            enabled.append(bool(flags.driver_enable_status))
        if any(value for key,value in vars(grip.msg.foc_status).items()
               if key not in ('driver_enable_status','homing_status')):
            raise RuntimeError('Gripper driver fault')
        if error or code not in ([0,1] if allow_estop else [0]):
            # This type is still fatal in all ordinary status checks. Only the
            # post-Reset, pre-motion wait may await clearance of a pure 0x05.
            if code == 5 and not error & ~0x003f:
                msg = status.msg
                fields = [int(getattr(msg,key)) for key in
                          ('ctrl_mode','arm_status','mode_feedback','teach_status',
                           'motion_status','trajectory_num')]
                details = dict(timestamp=status.timestamp,arm_status=code,err_code=error,
                               communication_joints=[i+1 for i in range(6) if error & (1<<i)],
                               status_can_hex=bytes(fields+[error>>8,error&255]).hex(),
                               enabled=enabled)
                raise JointCommunicationFault('Arm protection/fault: '+str(msg),details)
            raise RuntimeError('Arm protection/fault: '+str(status.msg))
        if require_enabled and (not all(enabled) or int(status.msg.ctrl_mode)!=1
                                or int(status.msg.mode_feedback)!=1):
            raise RuntimeError('CAN/MOVEJ mode or joint enable feedback lost')
        return code, enabled

    def joints(self, values):
        self.guard.check()
        values=np.asarray(values,dtype=float)
        if values.shape!=(6,) or not np.isfinite(values).all():raise RuntimeError('Invalid joint command')
        if np.any(values<self._policy_lower-1e-8) or np.any(values>self._policy_upper+1e-8):
            raise RuntimeError('Final joint command outside live firmware limits')
        now=time.monotonic()
        if self._last_joint_send is not None and now-self._last_joint_send>.1:
            raise RuntimeError('Joint command stream gap exceeds 100ms; refuse to resume')
        # Match the working collector: latched mode is also refreshed at 10Hz.
        # The controller owns this clock; no background thread continues motion
        # commands after a stalled control loop.
        if now-self._last_mode>=.1:
            self.arm._msg_mode.move_spd_rate_ctrl = 5 if self._control_profile=='prepare' else 100
            self.arm.set_motion_mode('j' if self._control_profile=='prepare' else 'js')
            self._last_mode=time.monotonic()
        move=self.arm.move_j if self._control_profile=='prepare' else self.arm.move_js
        move(np.asarray(values,dtype=float).tolist())
        self._last_joint_send=time.monotonic()

    def configure_follower(self):
        self.guard.check()
        self.arm.set_follower_mode()  # Working collector's 0x470 FC 00 ...

    def set_control_profile(self, profile):
        if profile not in ('prepare','policy'):raise ValueError('Unknown control profile')
        self.guard.check()
        if profile!=self._control_profile:
            self._control_profile=profile
            self._last_mode=float('-inf')

    def grip(self, width, force):
        self.guard.check()
        self.gripper.move_gripper_m(value=float(np.clip(width,0,.1)),force=float(force))

    def close(self):
        self.closed.set()
        if self.thread: self.thread.join(.5)
        self.arm.disconnect()  # Closes CAN threads only; no disable command.
        if self.guard.armed.value:
            try:
                root=Path(__file__).resolve().parents[1]/'logs'
                root.mkdir(exist_ok=True)
                record=dict(time=time.time(),channel='can1',
                            note='TX means queued successfully, not hardware acknowledgement',
                            frames=list(self._can_trace))
                content=json.dumps(record,separators=(',',':'))+'\n'
                name='can_trace_%s.json'%time.strftime('%Y%m%d_%H%M%S')
                (root/name).write_text(content)
                (root/'last_can_trace.json').write_text(content)
                print('CAN发送/反馈记录：',root/name,flush=True)
            except OSError as error:print('CAN记录写入失败：',error,flush=True)

    def _trace(self, direction, message):
        cid=message.arbitration_id
        if (0x150<=cid<=0x159 or cid in (0x470,0x471,0x2a1)
                or 0x261<=cid<=0x266 or 0x2a5<=cid<=0x2a8):
            self._can_trace.append(dict(t=time.monotonic(),direction=direction,
                                        can_id=hex(cid),data=bytes(message.data).hex()))
