"""Conservative leader geometry, held follower and additional damping tests."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from rlt_piper.collision import Model,overlap
from rlt_piper.dual_safety import leader_supervisor
from rlt_piper.leader_align import Aligner,software_version

MODEL=Path(__file__).resolve().parents[1]/'rlt_piper/leader_geometry.json'
if not MODEL.is_file():MODEL=Path(__file__).resolve().parent/'rlt_piper/leader_geometry.json'
class AlignmentTests(unittest.TestCase):
    def test_inflated_official_teaching_tool_refuses_folded_pose(self):
        model=Model(MODEL);q=np.radians([-3.92,.22,0,7.55,13.86,-16.82])
        self.assertIn('leader_tool',model.collision(q))
        with self.assertRaisesRegex(RuntimeError,'goal collision'):model.plan(np.radians([0,30,-60,0,30,0]),q)
    def test_all_points_of_a_valid_path_are_checked(self):
        model=Model(MODEL);start=np.radians([0,30,-60,0,30,0]);end=np.radians([10,40,-65,0,35,5])
        path=model.plan(start,end);self.assertEqual(len(path),2)
        for a,b in zip(path,path[1:]):self.assertTrue(model.edge(a,b))
    def test_safe_endpoints_do_not_allow_a_colliding_middle(self):
        model=Model(MODEL)
        a=np.radians([-144.997,44.415,-23.585,-85.422,33.942,-85.210])
        b=np.radians([107.626,60.714,-35.079,-25.657,16.836,80.739])
        self.assertIsNone(model.collision(a));self.assertIsNone(model.collision(b))
        self.assertIsNotNone(model.collision((a+b)/2));self.assertFalse(model.edge(a,b))
    def test_obb_detects_contact_and_resolves_separated_rotations(self):
        a=(np.zeros(3),np.eye(3),np.ones(3))
        self.assertTrue(overlap(a,(np.array([2.,0,0]),np.eye(3),np.ones(3))))
        self.assertFalse(overlap(a,(np.array([3.,0,0]),np.eye(3),np.ones(3))))
    def test_attachment_exclusion_must_be_proved_not_merely_configured(self):
        data=json.loads(MODEL.read_text());data['tool']['min'][2]=-.1
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'m.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError,'half-space'):Model(path)
    def test_cancelled_planning_cannot_produce_motion_path(self):
        model=Model(MODEL);model.cancelled=lambda:True
        with self.assertRaisesRegex(RuntimeError,'cancelled'):model.plan(np.zeros(6),np.zeros(6))
    def test_powered_leader_is_damped_on_shared_stop_without_disable(self):
        v=lambda n:SimpleNamespace(value=n)
        with patch('rlt_piper.dual_safety.damped_stop') as stop:
            leader_supervisor(1,v(1),v(1),v(0),v(0));stop.assert_called_once_with('can0')
        with patch('rlt_piper.dual_safety.damped_stop') as stop:
            leader_supervisor(1,v(0),v(1),v(0),v(0));stop.assert_not_called()
    def test_non_approved_packets_cannot_actuate_handle_or_disable(self):
        a=Aligner.__new__(Aligner);a.guard=SimpleNamespace(stop=SimpleNamespace(value=0))
        for cid in [0x150,0x159,0x471]:
            with self.assertRaisesRegex(RuntimeError,'Unapproved'):a.send(cid,bytes(8))
    def test_stop_latch_is_checked_before_every_can_send(self):
        a=Aligner.__new__(Aligner);a.guard=SimpleNamespace(stop=SimpleNamespace(value=1))
        with self.assertRaisesRegex(RuntimeError,'Stop latched'):a.send(0x155,bytes(8))
    def test_small_measured_boundary_error_does_not_expand_command_range(self):
        a=Aligner(None,SimpleNamespace(stop=SimpleNamespace(value=0)))
        q=np.radians([0,-.024,-60,0,30,0]);bounded=a.measured_start(q)
        self.assertEqual(bounded[1],0);self.assertEqual(a.model.collision(q),'joint limits');self.assertIsNone(a.model.collision(bounded))
        a.send=lambda *_:self.fail('Invalid target sent')
        with self.assertRaisesRegex(RuntimeError,'指令超限'):a.joints(q)
        q[1]=np.radians(-.101)
        with self.assertRaisesRegex(RuntimeError,'joint limits'):a.measured_start(q)
    def test_measured_boundary_tolerance_does_not_bypass_tool_collision(self):
        a=Aligner(None,SimpleNamespace(stop=SimpleNamespace(value=0)))
        q=np.radians([9.535,-.024,-.184,1.095,30.448,-31.161])
        with self.assertRaisesRegex(RuntimeError,'base/leader_tool'):a.measured_start(q)


class ProtocolSimulationTests(unittest.TestCase):
    def simulate(self,version='V1.8.9'):
        import struct,sys
        clock=SimpleNamespace(t=10.)
        clock.monotonic=lambda:clock.t
        clock.sleep=lambda seconds:setattr(clock,'t',clock.t+seconds)
        packets=[];targets=[]
        class Leader:
            def __init__(self):self.q=np.radians([0,30,-60,0,30,0]);self.desired=self.q.copy();self.updated=clock.t;self.error=None;self.role=0xFA;self.query=1;self.bus=self;self.native=False
            def advance(self):
                self.q=self.desired+(self.q-self.desired)*np.exp(-(clock.t-self.updated)/.06);self.updated=clock.t
            def vector(self):
                if not self.native:raise RuntimeError('Native teaching commands silent until drag restored')
                if self.role!=0xFA:raise RuntimeError('Native frames unavailable in controlled mode')
                self.advance()
                return np.r_[self.q,.04],np.full(4,clock.t+.0001)
            def feedback(self):self.advance();return np.r_[self.q,0.],np.full(3,clock.t+.0001)
            def status(self):
                if self.role!=0xFC:raise RuntimeError('Not in controlled mode')
            def since(self,cid,after):
                if cid==0x4AF:
                    if isinstance(version,bytes):raw=version
                    else:
                        import re
                        match=re.fullmatch(r'V(\d)\.(\d)\.(\d)',version)
                        if match:
                            raw=bytearray(88);raw[:8]=b'H-V1.2-1';raw[60:68]=('S-V%s.%s-%s'%match.groups()).encode();raw[68:74]=b'260916';raw=bytes(raw)
                        else:raw=version.encode()
                    return [(raw[i:i+8],after+.0001+i*.00001) for i in range(0,len(raw),8)]
                model=Model(MODEL);j=self.query-1
                return [(bytes([self.query])+struct.pack('>hhH',int(np.degrees(model.upper[j])*10),int(np.degrees(model.lower[j])*10),1000)+b'\0',after+.0001)]
            def send(self,message,timeout):
                cid,data=message.arbitration_id,message.data;packets.append((cid,data))
                if cid==0x470:self.role=data[0]
                elif cid==0x191:self.native=True
                elif cid==0x472:self.query=data[0]
                elif cid in (0x155,0x156,0x157):
                    self.advance();i=(cid-0x155)*2;self.desired[i:i+2]=np.radians(np.array(struct.unpack('>ii',data))/1000)
                    if cid==0x157:targets.append((clock.t,self.desired.copy()))
        leader=Leader();guard=SimpleNamespace(stop=SimpleNamespace(value=0),leader_active=SimpleNamespace(value=0))
        align=Aligner(leader,guard);align.target=np.r_[np.radians([2,31,-61,0,30,0]),.02]
        with patch('rlt_piper.leader_align.time',clock),patch.dict(sys.modules,{'can':SimpleNamespace(Message=lambda **kw:SimpleNamespace(**kw))}):align.run()
        return align,guard,packets,targets
    def test_complete_switch_align_restore_uses_no_disable_reset_or_gripper(self):
        align,guard,packets,targets=self.simulate()
        self.assertIsNone(align.error);self.assertIsNotNone(align.result);self.assertEqual(align.phase,'ready')
        self.assertFalse(guard.stop.value);self.assertTrue(guard.leader_active.value)
        self.assertFalse({0x150,0x159,0x471}&{cid for cid,_ in packets})
        roles=[data[0] for cid,data in packets if cid==0x470];self.assertEqual(roles,[0xFC,0xFA])
        self.assertEqual(packets[-1],(0x191,bytes([1,0,0,0,0,0,0,0])))
        for (ta,a),(tb,b) in zip(targets,targets[1:]):
            if tb>ta:self.assertLessEqual(float(np.degrees(np.abs(b-a)).max())/(tb-ta),5.05)
    def test_build_date_cannot_be_mistaken_for_a_supported_firmware(self):
        align,guard,packets,_=self.simulate('V1.8.4 build 2026.11.13')
        self.assertIsNotNone(align.error);self.assertEqual([cid for cid,_ in packets],[0x4AF])
    def test_old_firmware_refused_before_any_mode_or_motion_packet(self):
        align,guard,packets,_=self.simulate('V1.8.4')
        self.assertIsNotNone(align.error);self.assertTrue(guard.stop.value);self.assertFalse(guard.leader_active.value)
        self.assertEqual([cid for cid,_ in packets],[0x4AF])
    def test_real_leader_response_ignores_hardware_version_and_build_date(self):
        raw=bytes.fromhex('482d56312e322d3100000000000000003130000000000000000000000000000050697065725f4c5f4d43000000000000333041480000000000000000532d56312e392d31323630393136000031350000de7b000000000000')
        self.assertEqual(software_version(raw),'S-V1.9-1')
        align,guard,packets,_=self.simulate(raw)
        self.assertIsNone(align.error);self.assertEqual(align.phase,'ready');self.assertFalse(guard.stop.value)
        self.assertFalse({0x150,0x159,0x471}&{cid for cid,_ in packets})
    def test_incomplete_or_malformed_response_refused_before_motion(self):
        for raw in [b'H-V1.9-1',bytes(88),b'H-V1.2-1'+bytes(80),b'H-V1.2-1'+bytes(80)+bytes(8)]:
            with self.subTest(raw=raw):
                align,guard,packets,_=self.simulate(raw)
                self.assertIsNotNone(align.error);self.assertTrue(guard.stop.value);self.assertEqual([cid for cid,_ in packets],[0x4AF])
    def test_new_hardware_cannot_override_old_software(self):
        raw=bytearray(88);raw[:8]=b'H-V1.9-9';raw[60:68]=b'S-V1.8-4';raw[68:74]=b'261116'
        align,guard,packets,_=self.simulate(bytes(raw))
        self.assertIsNotNone(align.error);self.assertEqual([cid for cid,_ in packets],[0x4AF])

if __name__=='__main__':unittest.main()
