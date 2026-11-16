"""Native silence is held with original timestamps and an independent ping."""
import struct,threading,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from rlt_piper.hil import Leader
from rtc_piper.observations import NotReady

class InputTests(unittest.TestCase):
    def setup_leader(self):
        clock=SimpleNamespace(t=10.);clock.monotonic=lambda:clock.t
        leader=Leader.__new__(Leader);leader.lock=threading.Lock();leader.frames={};leader.error=None
        leader.cfg=dict(leader_gripper_closed_raw=0,gripper_max_m=.1);leader.partial_since=None
        leader.held_native=(np.r_[np.zeros(6),.02],np.full(4,9.))
        sent=[];leader.bus=SimpleNamespace(send=lambda msg,timeout:sent.append(msg))
        return leader,clock,sent
    def patches(self,clock):
        import contextlib,sys
        stack=contextlib.ExitStack();stack.enter_context(patch('rlt_piper.hil.time',clock))
        stack.enter_context(patch.dict(sys.modules,{'can':SimpleNamespace(Message=lambda **kw:SimpleNamespace(**kw))}));return stack
    def test_silent_input_retains_pose_and_original_timestamps(self):
        leader,clock,sent=self.setup_leader()
        with self.patches(clock):
            for _ in range(20):
                leader.frames[0x4AF]=(bytes(8),clock.t);q,ts=leader.following();clock.t+=.05
                np.testing.assert_array_equal(ts,np.full(4,9.));self.assertEqual(q[6],.02)
        self.assertTrue(sent);self.assertTrue(all(m.arbitration_id==0x4AF and m.data==bytes([1,0,0,0,0,0,0,0]) for m in sent))
    def test_lost_link_cannot_be_hidden_by_cached_input(self):
        leader,clock,_=self.setup_leader()
        with self.patches(clock):
            leader.following();clock.t+=.501
            with self.assertRaisesRegex(RuntimeError,'500ms'):leader.following()
    def test_partial_input_is_not_accepted_as_a_new_complete_sample(self):
        leader,clock,_=self.setup_leader();leader.frames[0x155]=(bytes(8),clock.t)
        with self.patches(clock):
            leader.following();clock.t+=.06;leader.frames[0x4AF]=(bytes(8),clock.t)
            with self.assertRaisesRegex(RuntimeError,'部分示教帧'):leader.following()
    def test_invalidating_input_prevents_reusing_pre_restore_pose(self):
        leader,clock,_=self.setup_leader()
        with self.patches(clock):
            leader.invalidate_native()
            with self.assertRaises(NotReady):leader.following()
    def test_actual_firmware_and_driver_faults_remain_fatal(self):
        for cid,data in [(0x2A1,bytes([2,1,1,0,0,0,0,0])),(0x261,bytes([0,0,0,0,0,1,0,0]))]:
            leader,clock,_=self.setup_leader();leader.frames[cid]=(data,clock.t)
            with self.patches(clock),self.assertRaisesRegex(RuntimeError,'故障'):leader.following()

if __name__=='__main__':unittest.main()
