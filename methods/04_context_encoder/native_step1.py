"""Official convolution-only Caffe feature release, explicitly selected at eval.

The grouped convolutions, LRN and pool5 implement the released AlexNet feature
architecture. This path has no decoder or learned fully-connected bottleneck.
"""
import torch
from torch import nn
from torchvision import transforms as T


class CaffePixels:
    """RGB [0,1] to BGR [0,255], subtracting the evaluated channel means."""
    def __call__(self, image):
        return image[[2, 1, 0]] * 255.0 - image.new_tensor([104., 117., 123.])[:, None, None]


def make_transform():
    return T.Compose([T.Resize(256), T.CenterCrop(227), T.ToTensor(), CaffePixels()])


class OfficialFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        # (input, output, kernel, stride, padding, groups, pool, LRN)
        for ci, co, kernel, stride, padding, groups, pool, lrn in (
            (3, 96, 11, 4, 0, 1, True, True),
            (96, 256, 5, 1, 2, 2, True, True),
            (256, 384, 3, 1, 1, 1, False, False),
            (384, 384, 3, 1, 1, 2, False, False),
            (384, 256, 3, 1, 1, 2, True, False),
        ):
            layers.extend([nn.Conv2d(ci, co, kernel, stride, padding, groups=groups), nn.ReLU(inplace=True)])
            if pool:
                layers.append(nn.MaxPool2d(3, 2))
            if lrn:
                layers.append(nn.LocalResponseNorm(5, alpha=1e-4, beta=.75, k=1.))
        self.features = nn.Sequential(*layers)

    def forward(self, image):
        # Preserve the evaluator's (reconstruction, features) interface.
        return None, self.features(image).flatten(1)


def load_encoder(state):
    model = OfficialFeatures()
    model.load_state_dict(state, strict=True)
    return model
