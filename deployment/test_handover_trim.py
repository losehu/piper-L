"""Waiting at handovers is cropped, task pauses and real clocks are preserved."""
import json,tempfile,unittest
from types import SimpleNamespace
import numpy as np
from rlt_piper.trimming import keep_commands
from rlt_piper.recording import Recorder,transitions

CFG=dict(fps=30,control_hz=120,execute_horizon=18,chunk_smoothing_steps=12,command_filter_tau_s=.05)
def trace(grip=False):
    t=np.arange(10.,20.001,.01);q=np.zeros((len(t),7))
    value=np.clip((t-12)/2,0,1)+np.clip((t-16)/2,0,1)
    q[:,6 if grip else 0]=value*(.005 if grip else .2)
    data=dict(command_t=t,commands=q,command_source=np.full(len(t),2),command_generation=np.full(len(t),2))
    meta=dict(events=[dict(mode='human',generation=2,t=10.),dict(mode='policy',generation=3,t=20.01)],live=False)
    return data,meta

class TrimTests(unittest.TestCase):
    def test_both_waiting_edges_trimmed_but_interior_grasp_pause_kept(self):
        d,m=trace();keep,report=keep_commands(d,m);t=d['command_t']
        self.assertFalse(keep[np.argmin(abs(t-11))]);self.assertFalse(keep[np.argmin(abs(t-19.5))])
        self.assertTrue(keep[np.argmin(abs(t-15))]);self.assertTrue(keep[np.argmin(abs(t-12.5))])
        self.assertGreater(report['excluded_seconds'],2.5)
    def test_gripper_only_correction_is_motion(self):
        d,m=trace(grip=True);keep,_=keep_commands(d,m);t=d['command_t']
        self.assertTrue(keep[np.argmin(abs(t-12.5))]);self.assertFalse(keep[np.argmin(abs(t-19.5))])
    def test_feedback_motion_kept_even_if_command_constant(self):
        d,m=trace();d['command_state']=d['commands'].copy();d['commands'][:]=0
        keep,_=keep_commands(d,m);t=d['command_t']
        self.assertTrue(keep[np.argmin(abs(t-13))]);self.assertFalse(keep[np.argmin(abs(t-19.5))])
    def test_live_tail_wait_is_withheld_then_interior_pause_returns(self):
        d,m=trace();m['events']=m['events'][:1];m['live']=True
        live={k:v[d['command_t']<=15.5] for k,v in d.items()};keep,_=keep_commands(live,m)
        self.assertFalse(keep[np.argmin(abs(live['command_t']-15))])
        keep,_=keep_commands(d,m);self.assertTrue(keep[np.argmin(abs(d['command_t']-15))])
    def test_live_wait_before_first_human_motion_is_never_published(self):
        d,m=trace();m['events']=m['events'][:1];m['live']=True
        live={k:v[d['command_t']<=11.9] for k,v in d.items()}
        keep,_=keep_commands(live,m);self.assertFalse(keep.any())
    def test_short_stops_and_final_task_reward_pause_not_blanket_deleted(self):
        d,m=trace();m['events']=m['events'][:1];m.update(live=False,labelled=True)
        keep,_=keep_commands(d,m);self.assertTrue(keep[np.argmin(abs(d['command_t']-19.5))])
    def test_final_static_context_is_bounded_without_dropping_success_reward(self):
        d,m=trace();d['commands'][:]=0;m['events']=m['events'][:1]
        m.update(live=False,labelled=True,terminal_available=True,label_at=20.)
        keep,_=keep_commands(d,m);self.assertTrue(keep.any())
        self.assertGreaterEqual(d['command_t'][keep][0],19.39)
        self.assertLess(np.count_nonzero(keep),65)
    def test_replay_does_not_join_clocks_or_bootstrap_into_trimmed_wait(self):
        d,m=trace()
        with tempfile.TemporaryDirectory() as folder:
            r=Recorder(folder,'online','token',CFG)
            for event in m['events']:r.event(event['mode'],event['generation'],event['t'])
            for t,q in zip(d['command_t'],d['commands']):r.command(t,q,'human',2,measured=q,state_times=np.full(4,t))
            for t in np.arange(10.,20.,.1):
                result=dict(generation=2,rlt=dict(z_rl=np.ones(512)*t,proprio=np.zeros(7),ref_chunk=np.zeros((18,7)),proposed_chunk=np.ones((18,7)),source='base',actor_version=0))
                r.plan(SimpleNamespace(t=t),result,t+.05,source='human')
            path=r.snapshot(live=True);rows=transitions(path,'token');self.assertTrue(rows)
            starts=[float(row['transition_id'].split(':')[-1]) for row in rows]
            self.assertGreater(min(starts),11.5);self.assertLess(max(starts)+.6,19.)
            self.assertTrue(any(14.5<t<15.5 for t in starts))
            self.assertTrue(all(not row['done'] and not row['rewards'].any() for row in rows))
            with np.load(path/json.loads((path/'episode.json').read_text())['trace']) as raw:
                self.assertEqual(len(raw['commands']),len(d['commands']));np.testing.assert_array_equal(raw['command_t'],d['command_t'])
            summary=json.loads((path/'episode.json').read_text())['training_trim']
            self.assertEqual(summary['clock'],'original_monotonic_no_concatenation');self.assertGreater(summary['excluded_commands'],0)

if __name__=='__main__':unittest.main()
