"""No robot/GPU: backend selection, preprocessing contracts and service ownership."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock
import numpy as np
from rtc_piper.cli import config
from rtc_piper.profiles import select_model
from rtc_piper.pi05_backend import preflight,stop_record
from realtime_vla_runtime.policy import compact_inputs
from realtime_vla_runtime.export import allocation_shapes

ROOT=Path(__file__).resolve().parent


class SelectionTests(unittest.TestCase):
    def test_backend_is_independent_of_checkpoint_rtc_and_smoothing(self):
        base=config(ROOT/'config.example.yaml')
        original=select_model(base,'pi05',ROOT)
        accelerated=select_model(base,'pi05',ROOT,pi05_backend='realtime-vla')
        self.assertEqual(original['inference_backend'],'jax')
        for key in ['model_step','checkpoint_index','checkpoint','execute_horizon','chunk_smoothing_steps',
                    'command_filter_tau_s','action_horizon','execution_mode']:
            self.assertEqual(original[key],accelerated[key])
        self.assertEqual(accelerated['policy_port'],8004)
        disabled=select_model(base,'pi05',ROOT,pi05_backend='realtime-vla',no_rtc=True,no_smoothing=True,pi05_step=1500)
        self.assertEqual((disabled['model_step'],disabled['execution_mode']),(1500,'sync_chunks'))
        self.assertFalse(disabled['smoothing_enabled'])
        with self.assertRaisesRegex(ValueError,'pi05 only'):
            select_model(base,'vpp2',ROOT,pi05_backend='realtime-vla')

    def test_missing_accelerator_assets_do_not_stop_any_existing_service(self):
        import rtc_piper.pi05_backend as b
        with tempfile.TemporaryDirectory() as d,mock.patch('os.kill') as kill:
            with self.assertRaisesRegex(RuntimeError,'missing'):
                b.ensure(Path(d),dict(model_step=2500,checkpoint_index=2499))
            kill.assert_not_called()

    def test_unvalidated_assets_are_refused_before_model_switch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['.realtime-vla-venv/bin/python','vendor/realtime_vla/pi05_infer.py',
                         'realtime_vla_assets/pi05_2500/model.safetensors']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            (root/'realtime_vla_assets/pi05_2500/metadata.json').write_text(json.dumps(dict(training_updates=2500,checkpoint_index=2499)))
            (root/'logs').mkdir();(root/'logs/realtime_vla_validation_2500.json').write_text(json.dumps(dict(status='failed',cases=6)))
            with self.assertRaisesRegex(RuntimeError,'validation failed'):
                preflight(root,dict(model_step=2500,checkpoint_index=2499))


class InputTests(unittest.TestCase):
    def values(self):
        return dict(image={key:np.full((224,224,3),value,dtype=np.uint8) for key,value in
                           [('base_0_rgb',0),('left_wrist_0_rgb',255),('right_wrist_0_rgb',125)]},
                    image_mask=dict(base_0_rgb=True,left_wrist_0_rgb=True,right_wrist_0_rgb=False),
                    tokenized_prompt=np.arange(200),tokenized_prompt_mask=np.arange(200)<93)
    def test_compaction_preserves_valid_tokens_and_original_pixel_normalization(self):
        images,ids=compact_inputs(self.values())
        self.assertEqual(images.shape,(2,224,224,3));np.testing.assert_array_equal(ids,np.arange(93))
        np.testing.assert_array_equal(images[0],-1);np.testing.assert_array_equal(images[1],1)
    def test_unexpected_camera_masks_and_noncontiguous_prompt_are_refused(self):
        values=self.values();values['image_mask']['right_wrist_0_rgb']=True
        with self.assertRaisesRegex(ValueError,'masked third'):compact_inputs(values)
        values=self.values();values['tokenized_prompt_mask'][1]=False
        with self.assertRaisesRegex(ValueError,'contiguous'):compact_inputs(values)
    def test_cpu_export_shape_table_includes_adaptive_norm_and_full_vocabulary(self):
        shapes=allocation_shapes(ROOT/'vendor/realtime_vla/convert_from_jax_pi05.py')
        self.assertEqual(shapes['embedding_weight'],(257152,2048))
        self.assertEqual(shapes['decoder_pre_attn_norm_mod_w'],(18,1024,3072))
        self.assertEqual(shapes['language_embeds'],(0,2048))


class OwnershipTests(unittest.TestCase):
    def test_connected_and_unknown_processes_remain_untouched(self):
        import rtc_piper.model_services as m
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'state.json';args=['python','-m','realtime_vla_runtime.serve','--root',str(root)]
            p.write_text(json.dumps(dict(pid=123,owner_root=str(root),argv=args)))
            with mock.patch.object(m,'process_args',return_value=['other-task']),mock.patch('os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'ownership'):stop_record(root,p,'realtime_vla_runtime.serve',8004)
                kill.assert_not_called()
            with mock.patch.object(m,'process_args',return_value=args),mock.patch.object(m,'connected',return_value=True),mock.patch('os.kill') as kill:
                with self.assertRaisesRegex(RuntimeError,'connected client'):stop_record(root,p,'realtime_vla_runtime.serve',8004)
                kill.assert_not_called()

    def test_client_rejects_wrong_backend_even_when_checkpoint_matches(self):
        from rtc_piper.policy import Client
        from openpi_client import msgpack_numpy
        cfg=select_model(config(ROOT/'config.example.yaml'),'pi05',ROOT,pi05_backend='realtime-vla')
        metadata=dict(model='pi05_piper_grasp_2500',training_updates=2500,source_checkpoint_index=2499,
                      action_horizon=50,physical_action_dim=7,fps=30,warmed_up=True,inference_backend='jax')
        ws=mock.Mock();ws.recv.return_value=msgpack_numpy.packb(metadata)
        with mock.patch('websockets.sync.client.connect',return_value=ws):
            with self.assertRaisesRegex(RuntimeError,'identity'):Client(cfg)
        ws.close.assert_called_once()

if __name__=='__main__':unittest.main()
