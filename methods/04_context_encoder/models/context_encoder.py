"""
Context Encoder: Feature Learning by Inpainting (Pathak et al., 2016).
Paper: https://arxiv.org/abs/1604.07379

Step 1 only: the AlexNet-based architecture. The step 2 ViT variant
(ContextEncoderViT) and the official Caffe feature extractor were not brought
across -- step 2 has no place in this port (the contract's stages are pretrain and
linear_eval), and dropping the ViT also drops its `timm` dependency. The
ContextEncoderAlexNet and Discriminator classes below are the captured code,
unchanged.
"""

import torch
import torch.nn as nn


class ContextEncoderAlexNet(nn.Module):
    """
    Original Context Encoder architecture based on AlexNet
    Used for Step 1: strict reproduction of the original paper
    """
    def __init__(self, channels=3):
        super().__init__()
        
        # Encoder: AlexNet-like convolutional layers
        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(channels, 96, kernel_size=11, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Adaptive pooling to ensure 7x7 output for any input size
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        
        # Channel-wise fully connected layer (bottleneck)
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Decoder: Transposed convolutions to output 128x128
        self.decoder_fc = nn.Linear(4096, 256 * 8 * 8)
        
        self.decoder = nn.Sequential(
            # Upsample from 8x8 to 16x16
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Upsample from 16x16 to 32x32
            nn.ConvTranspose2d(256, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            
            # Upsample from 32x32 to 64x64
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            # Final layer: 64x64 to 128x128
            nn.ConvTranspose2d(96, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        batch_size = x.size(0)
        
        # Flatten and pass through FC layers
        x = x.view(batch_size, -1)
        features = self.fc(x)
        
        # Decode
        x = self.decoder_fc(features)
        x = x.view(batch_size, 256, 8, 8)
        x = self.decoder(x)
        
        return x, features


class Discriminator(nn.Module):
    """
    Discriminator network for adversarial training
    Using InstanceNorm instead of BatchNorm to avoid version conflicts
    """
    def __init__(self, channels=3, img_size=128):
        super().__init__()
        if img_size < 16:
            raise ValueError('discriminator input must be at least 16x16')

        layers = []
        spatial = img_size
        in_features = channels
        channel_schedule = (64, 128, 256, 512, 512)
        stage = 0
        while spatial > 4 and spatial % 2 == 0 and stage < len(channel_schedule):
            out_features = channel_schedule[stage]
            layers.append(nn.Conv2d(in_features, out_features, 4, 2, 1))
            if stage > 0:
                layers.append(nn.InstanceNorm2d(out_features, affine=True))
            layers.append(nn.LeakyReLU(0.2))
            spatial //= 2
            in_features = out_features
            stage += 1
        if stage == 0:
            raise ValueError(f'unsupported discriminator image size: {img_size}')

        # Pathak's center discriminator sees only real or generated hole pixels,
        # never context or a mask, and returns one real/fake logit per image.
        layers.append(nn.Conv2d(in_features, 1, kernel_size=spatial, stride=1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        validity = self.model(x)
        return validity.view(-1, 1)


def create_model(model_type='alexnet', **kwargs):
    """Factory for the Context Encoder pretrain model.

    Only 'alexnet' is available here. 'vit' belonged to step 2, which this port
    does not include (its model needed `timm` and was not brought across); it is
    refused by name rather than left to fail as an AttributeError elsewhere.
    """
    if model_type == 'alexnet':
        return ContextEncoderAlexNet(**kwargs)
    if model_type in ('vit', 'official_alexnet'):
        raise ValueError(
            f"model_type={model_type!r} belongs to step 2 / the official Caffe "
            "feature path, which this port does not include; only 'alexnet' is "
            "available")
    raise ValueError(f"Unknown model type: {model_type}")
