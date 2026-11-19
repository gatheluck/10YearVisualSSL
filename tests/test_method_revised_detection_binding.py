"""Native encoder pyramid capability for revised detector construction."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class RevisedBinding(unittest.TestCase):
    def test_native_video_encoder_pyramid(self):
        from torch import nn

        from downstream.revised_detection import _binding
        from methods.vjepa2_1.downstream_backbone import Backbone
        class Encoder(nn.Module):
            embed_dim = 4
        encoder=Encoder()
        body=Backbone(encoder)
        self.assertEqual(_binding(body),([.485,.456,.406],[.229,.224,.225],'forward_features'))
