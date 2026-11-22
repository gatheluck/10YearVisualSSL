"""Selected Figure 3 providers, geometry and training schedules."""
import json

from tests.test_figure3_lora_configs import CampaignFixture


class SelectedProviderTests(CampaignFixture):
    def test_complete_source_selected_matrix(self):
        self.assertEqual(set(self.models), {'dinov3_hf', 'raev2_k7', 'vggt_omega', 'vjepa2_1', 'siglip2_g'})
        p = self.invoke()
        self.assertEqual(p.returncode, 0, p.stderr)
        files = {f.name for f in self.out.glob('*')}
        self.assertEqual(files, {f'{t}-{m}-s{s}.json' for m in self.models
                                 for t in self.tasks for s in range(3)})
        for model in self.models:
            for task in self.tasks:
                for seed in range(3):
                    cfg = json.loads((self.out / f'{task}-{model}-s{seed}.json').read_bytes())
                    expected = {
                        'profile': 'b5x_lora_dpt_v2', 'task': task, 'track': 'lora',
                        'seed': seed, 'device': 'cuda', 'data_root': self.paths['datasets'][task],
                        'backbone': {'kind': model, 'arch': 'vit_giant_xformers'
                                     if model == 'vjepa2_1' else 'released',
                                     'img_size': 384 if model in ('vjepa2_1', 'siglip2_g') else 224,
                                     'patch_size': 16, 'encoder': self.paths['encoders'][model]},
                        'continuation': True, 'worker_lifecycle': 'reseed_each_epoch',
                        'workers': 6 if task == 'coco' else 10,
                    }
                    if task == 'ssv2':
                        expected['micro_batch_size'] = 8
                    if model == 'dinov3_hf' and task != 'coco':
                        epochs = 50 if task == 'imagenet' else 24
                        expected['schedule'] = f'B5X_LORA_LP_{epochs}EP_v2'
                    self.assertEqual(cfg, expected)
