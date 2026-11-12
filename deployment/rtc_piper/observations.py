"""Native RealSense images and capture-aligned SDK feedback, without ROS."""
from collections import deque
from dataclasses import dataclass
import threading
import time

import numpy as np


class NotReady(RuntimeError):
    def __init__(self,message,details=None):
        super().__init__(message)
        self.details=details


@dataclass
class Observation:
    t: float
    state: np.ndarray
    images: dict
    state_times: np.ndarray


class CaptureBank:
    def __init__(self):
        self.lock = threading.Lock()
        self.states = {key:deque(maxlen=256) for key in ['j12','j34','j56','grip']}
        self.images = {key:deque(maxlen=16) for key in ['cam0','cam1']}

    def state(self, key, stamp, values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(stamp) or not np.isfinite(values).all():
            raise RuntimeError('Nonfinite SDK feedback')
        with self.lock:
            history = self.states[key]
            if history and stamp < history[-1][0]:
                raise RuntimeError('SDK feedback timestamp moved backwards')
            if not history or stamp > history[-1][0]:
                history.append((stamp, values.copy()))

    def image(self, key, stamp, arrived, rgb):
        with self.lock:
            self.images[key].append((stamp, arrived, rgb))

    def live(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            rows = [values[-1] if values else None for values in self.states.values()]
        if any(row is None or not -.01 <= now-row[0] <= .15 for row in rows):
            raise NotReady('Waiting for fresh six-joint and gripper feedback')
        return np.concatenate([row[1] for row in rows]), np.array([row[0] for row in rows])

    def observation(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            images = {key:[frame for frame in values
                           if -.01 <= now-frame[0] <= .2 and 0 <= now-frame[1] <= .2]
                      for key,values in self.images.items()}
            pairs = [(a,b) for a in images['cam0'] for b in images['cam1']
                     if abs(a[0]-b[0]) <= .025]
            if not pairs:
                raise NotReady('Waiting for a fresh camera pair within 25ms',self._diagnostics(now))
            a,b = max(pairs, key=lambda p:(min(p[0][0],p[1][0]),-abs(p[0][0]-p[1][0])))
            rows = [min(values, key=lambda v:abs(v[0]-a[0]), default=None)
                    for values in self.states.values()]
        if any(row is None or abs(row[0]-a[0]) > .05 for row in rows):
            raise NotReady('No measured state within 50ms of image capture',self.diagnostics(now))
        state = np.concatenate([row[1] for row in rows])
        return Observation(a[0], state, dict(cam0=a[2],cam1=b[2]),
                           np.array([row[0] for row in rows]))

    def _diagnostics(self,now):
        images={name:dict(buffered=len(values),
                         capture_age_ms=1000*(now-values[-1][0]) if values else None,
                         arrival_age_ms=1000*(now-values[-1][1]) if values else None)
                for name,values in self.images.items()}
        recent={name:[v for v in values if -.01<=now-v[0]<=.2 and 0<=now-v[1]<=.2]
                for name,values in self.images.items()}
        skew=min((abs(a[0]-b[0]) for a in recent['cam0'] for b in recent['cam1']),default=None)
        return dict(images=images,min_fresh_pair_skew_ms=skew*1000 if skew is not None else None,
                    state_age_ms={name:1000*(now-values[-1][0]) if values else None
                                  for name,values in self.states.items()})

    def diagnostics(self,now=None):
        with self.lock:return self._diagnostics(time.monotonic() if now is None else now)


class RealSense:
    def __init__(self, config, bank):
        self.config, self.bank = config, bank
        self.stop = threading.Event()
        self.errors = []
        self.stats_lock=threading.Lock()
        self.stats={name:dict(received=0,accepted=0,rejected_domain=0,rejected_age=0,
                              last_domain=None,last_arrival=None,last_capture=None,
                              max_arrival_gap_ms=0.,usb_type=None)
                    for name in config['cameras']}
        self.threads = [threading.Thread(target=self._run,args=(name,serial),daemon=True)
                        for name,serial in config['cameras'].items()]
        for thread in self.threads:
            thread.start()

    def _run(self, name, serial):
        pipe = None
        try:
            import pyrealsense2 as rs
            pipe, config = rs.pipeline(), rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color,640,480,rs.format.rgb8,30)
            profile = pipe.start(config)
            device=profile.get_device()
            with self.stats_lock:
                self.stats[name]['usb_type']=device.get_info(rs.camera_info.usb_type_descriptor)
            for sensor in profile.get_device().query_sensors():
                if sensor.get_info(rs.camera_info.name) != 'RGB Camera':
                    continue
                for option,value in [(rs.option.global_time_enabled,1),
                                     (rs.option.frames_queue_size,1),
                                     (rs.option.auto_exposure_priority,0)]:
                    if sensor.supports(option): sensor.set_option(option,value)
            while not self.stop.is_set():
                frame = pipe.wait_for_frames(1000).get_color_frame()
                if not frame: continue
                now = time.monotonic()
                stamp = frame.get_timestamp()/1000-(time.time()-now)
                domain=str(frame.get_frame_timestamp_domain())
                valid_domain=domain=='timestamp_domain.global_time'
                valid_age=-.01<=now-stamp<=.2
                with self.stats_lock:
                    info=self.stats[name];info['received']+=1
                    if info['last_arrival'] is not None:
                        info['max_arrival_gap_ms']=max(info['max_arrival_gap_ms'],1000*(now-info['last_arrival']))
                    info.update(last_arrival=now,last_capture=stamp,last_domain=domain)
                    if not valid_domain:info['rejected_domain']+=1
                    elif not valid_age:info['rejected_age']+=1
                    else:info['accepted']+=1
                if not valid_domain or not valid_age:continue
                self.bank.image(name,stamp,now,np.asanyarray(frame.get_data()).copy())
        except Exception as error:
            self.errors.append('%s (%s): %s'%(name,serial,error))
        finally:
            if pipe is not None:
                try: pipe.stop()
                except Exception: pass

    def check(self):
        if self.errors:
            raise RuntimeError('Camera failure: '+str(self.errors))

    def diagnostics(self):
        with self.stats_lock:return {name:dict(info) for name,info in self.stats.items()}

    def close(self):
        self.stop.set()
        for thread in self.threads: thread.join(1.5)
