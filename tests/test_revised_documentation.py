"""Execute the documented task configuration through the real CLI validator."""

import importlib.util
import json
import re
import unittest
from pathlib import Path

HAVE_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE_TORCH, 'torch required for the task CLI validator')
class RevisedDocumentationTests(unittest.TestCase):
    def test_public_entry_docs_link_to_revised_cli(self):
        root=Path(__file__).resolve().parents[1]
        target=(root/'docs/CVPR_ADAPTATION.md').resolve()
        for entry in (root/'README.md',root/'docs/DOWNSTREAM.md'):
            links=re.findall(r'\[[^\]]+\]\(([^)]+)\)',entry.read_text())
            matching=[link for link in links if (entry.parent/link.split('#')[0]).resolve()==target]
            self.assertTrue(any(link.endswith('#revised-image-and-semantic-task-cli') for link in matching),
                            entry.name+' must link directly to the revised task CLI section')
            self.assertTrue(target.is_file())

    def test_documented_recipe_table_matches_resolved_source_routes(self):
        from downstream.revised_recipe import resolve_recipe
        text = (Path(__file__).resolve().parents[1] / 'docs/CVPR_ADAPTATION.md').read_text()
        rows = re.findall(r'^\| `([^`]+)` \| `([^`]+)` \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+|None) \| (\d+) \|$', text, re.MULTILINE)
        self.assertEqual({(row[0],row[1]) for row in rows}, {
            ('imagenet','lora'),('ade20k','lora'),('ade20k','dpt'),
            ('ssv2','lora'),('coco','lora'),('nyuv2','lora'),('nyuv2','dpt')})
        for task,track,epochs,batch,head,adapter,warm in rows:
            recipe=resolve_recipe(task,track)
            self.assertEqual((int(epochs),int(batch),float(head),None if adapter=='None' else float(adapter),int(warm)),
                             (recipe.epochs,recipe.effective_batch,recipe.head_lr,recipe.adapter_lr,recipe.warmup_epochs))

    def test_configuration_template_accepts_each_documented_route(self):
        from downstream.revised_tasks import validate_config

        doc = Path(__file__).resolve().parents[1] / 'docs/CVPR_ADAPTATION.md'
        blocks = re.findall(r'^```json\n(.*?)\n```', doc.read_text(), re.MULTILINE | re.DOTALL)
        self.assertEqual(len(blocks), 1, 'one complete CLI configuration template')
        config = json.loads(blocks[0])
        validate_config(config)
        for task, track in [('imagenet', 'lora'), ('ade20k', 'lora'), ('ade20k', 'dpt'),
                            ('ssv2', 'lora'), ('coco', 'lora'), ('nyuv2', 'lora'), ('nyuv2', 'dpt')]:
            with self.subTest(task=task, track=track):
                variant = dict(config, task=task, track=track)
                validate_config(variant)
                variant['diagnostic'] = {'epochs': 1, 'batch_size': 2, 'max_steps': 3}
                validate_config(variant)
                variant['epochs'] = 1
                with self.assertRaises(ValueError):
                    validate_config(variant)


if __name__ == '__main__':
    unittest.main()
