"""Execute the documented task configuration through the real CLI validator."""

import importlib.util
import json
import re
import unittest
from pathlib import Path

HAVE_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAVE_TORCH, 'torch required for the task CLI validator')
class RevisedDocumentationTests(unittest.TestCase):
    def test_configuration_template_accepts_each_documented_route(self):
        from downstream.revised_tasks import validate_config

        doc = Path(__file__).resolve().parents[1] / 'docs/CVPR_ADAPTATION.md'
        blocks = re.findall(r'^```json\n(.*?)\n```', doc.read_text(), re.MULTILINE | re.DOTALL)
        self.assertEqual(len(blocks), 1, 'one complete CLI configuration template')
        config = json.loads(blocks[0])
        validate_config(config)
        for task, track in [('imagenet', 'lora'), ('ade20k', 'lora'), ('ade20k', 'dpt')]:
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
