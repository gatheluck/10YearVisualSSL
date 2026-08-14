"""ViT Step-2 context-encoder model: a ViT-B/16 encoder + a transformer decoder.

Faithful to the capture's `ContextEncoderViT`: the centre-hole inpainting task on
a Vision Transformer. The encoder sees the image with the exact centred hole
pixels zeroed; a 4-layer transformer decoder, given the encoder memory and
learned mask tokens for the patches that overlap the hole, predicts the hole's
pixels patch-by-patch. Training adds a centre-hole adversarial discriminator
(the shared `Discriminator`, sized to the hole).

The capture builds the encoder with `timm.create_model('vit_base_patch16_224')`,
which fixes 224/768/12. This port builds the encoder as a `VisionTransformer`
directly (as the other ViT Step-2 ports do), so the dimensions are configurable
and a tiny model can run a CPU smoke; with the shipped config it is the same
ViT-B/16. This port's convention: the ViT trunk lives under ``self.encoder`` so
`encoder.pt` keeps only ``encoder.*`` (the decoder, prediction head, mask token,
decoder position embedding and the two mask buffers are pretext machinery,
excluded). ``get_features(x)`` returns the mean patch-token feature (embed_dim)
of the *unmasked* image -- the representation the linear probe reads. timm is
imported at module top, so this file (and thus timm) loads only on `arch: vit`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer

try:                                    # reuse the one Discriminator implementation
    from .context_encoder import Discriminator
except ImportError:                     # loaded as a top-level module in tests
    from models.context_encoder import Discriminator

__all__ = ["ContextEncoderViT", "Discriminator", "build_vit_context_encoder"]


class ContextEncoderViT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 decoder_dim: int = 512, decoder_depth: int = 4,
                 decoder_heads: int = 8, hole_size: int = 112) -> None:
        super().__init__()
        if image_size % patch_size or hole_size % patch_size:
            raise ValueError("image and hole sizes must be patch aligned")
        if not 0 < hole_size < image_size:
            raise ValueError("hole size must be smaller than the input image")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.hole_size = hole_size
        self.grid_size = image_size // patch_size
        self.hole_start = (image_size - hole_size) // 2
        self.hole_end = self.hole_start + hole_size
        num_patches = self.grid_size ** 2

        self.encoder = VisionTransformer(
            img_size=image_size, patch_size=patch_size, in_chans=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, num_classes=0, global_pool="",
            qkv_bias=True, norm_layer=nn.LayerNorm)

        pixel_mask = torch.zeros(1, 1, image_size, image_size, dtype=torch.bool)
        pixel_mask[:, :, self.hole_start:self.hole_end,
                   self.hole_start:self.hole_end] = True
        self.register_buffer("center_mask", pixel_mask, persistent=True)

        patch_starts = torch.arange(self.grid_size) * patch_size
        patch_ends = patch_starts + patch_size
        overlaps = (patch_starts < self.hole_end) & (patch_ends > self.hole_start)
        decoder_mask = overlaps[:, None] & overlaps[None, :]
        self.register_buffer("decoder_hole_mask",
                             decoder_mask.reshape(1, num_patches),
                             persistent=True)

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim, bias=True)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim, nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4, dropout=0.1, activation="gelu",
            batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                             num_layers=decoder_depth)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_dim))
        self.decoder_pred = nn.Linear(decoder_dim,
                                      patch_size ** 2 * in_channels, bias=True)
        self._init_decoder()

    def _init_decoder(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        for m in (self.decoder_embed, self.decoder, self.decoder_pred):
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.patch_size, self.in_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def extract_predicted_hole(self, pred: torch.Tensor) -> torch.Tensor:
        image = self.unpatchify(pred)
        return image[:, :, self.hole_start:self.hole_end,
                     self.hole_start:self.hole_end]

    def extract_target_hole(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :, self.hole_start:self.hole_end,
                      self.hole_start:self.hole_end]

    def forward_encoder(self, x: torch.Tensor):
        pixel_mask = self.center_mask.to(device=x.device).expand(
            x.shape[0], -1, -1, -1)
        masked = x.masked_fill(pixel_mask, 0)
        tokens = self.encoder.patch_embed(masked)
        cls = self.encoder.cls_token.expand(tokens.shape[0], -1, -1)
        latent = torch.cat([cls, tokens], dim=1) + self.encoder.pos_embed.to(
            device=tokens.device, dtype=tokens.dtype)
        latent = self.encoder.pos_drop(latent)
        if hasattr(self.encoder, "patch_drop"):
            latent = self.encoder.patch_drop(latent)
        if hasattr(self.encoder, "norm_pre"):
            latent = self.encoder.norm_pre(latent)
        for block in self.encoder.blocks:
            latent = block(latent)
        latent = self.encoder.norm(latent)
        return latent, pixel_mask

    def forward(self, x: torch.Tensor):
        latent, pixel_mask = self.forward_encoder(x)
        projected = self.decoder_embed(latent)
        pos = self.decoder_pos_embed.to(device=projected.device,
                                        dtype=projected.dtype)
        memory = projected + pos
        patch_queries = projected[:, 1:].clone()
        decoder_mask = self.decoder_hole_mask.to(device=x.device).expand(
            x.shape[0], -1)
        mask_tokens = self.mask_token.expand(patch_queries.shape[0],
                                             patch_queries.shape[1], -1)
        patch_queries = torch.where(decoder_mask.unsqueeze(-1), mask_tokens,
                                    patch_queries)
        queries = torch.cat([projected[:, :1], patch_queries], dim=1) + pos
        decoded = self.decoder(queries, memory)
        pred = self.decoder_pred(decoded[:, 1:])
        return pred, pixel_mask, latent[:, 1:]

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """The mean patch-token feature (embed_dim) of the *unmasked* image --
        the linear-probe representation. Uses the plain ViT (no hole mask)."""
        return self.encoder.forward_features(x)[:, 1:].mean(dim=1)


def build_vit_context_encoder(image_size: int = 224, patch_size: int = 16,
                              in_channels: int = 3, embed_dim: int = 768,
                              depth: int = 12, num_heads: int = 12,
                              mlp_ratio: float = 4.0, decoder_dim: int = 512,
                              decoder_depth: int = 4, decoder_heads: int = 8,
                              hole_size: int = 112) -> ContextEncoderViT:
    return ContextEncoderViT(
        image_size=image_size, patch_size=patch_size, in_channels=in_channels,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads,
        mlp_ratio=mlp_ratio, decoder_dim=decoder_dim, decoder_depth=decoder_depth,
        decoder_heads=decoder_heads, hole_size=hole_size)
