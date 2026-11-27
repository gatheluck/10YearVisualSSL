"""ViT backbones aligned with the released DINOv3 token semantics."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    keep = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
    return x * keep / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class RopePositionEmbedding(nn.Module):
    """Axial 2-D RoPE with the released coordinate parameterization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        normalize_coords: str = "separate",
        rescale_coords: float | None = 2.0,
    ):
        super().__init__()
        if embed_dim % (4 * num_heads):
            raise ValueError("embed_dim must be divisible by 4 * num_heads")
        self.head_dim = embed_dim // num_heads
        self.normalize_coords = normalize_coords
        self.rescale_coords = rescale_coords
        periods = base ** (
            2
            * torch.arange(self.head_dim // 4, dtype=torch.float32)
            / (self.head_dim // 2)
        )
        # Persistent because the teacher is initialized from the student state.
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.periods.device
        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, height, device=device) / height
            coords_w = torch.arange(0.5, width, device=device) / width
        elif self.normalize_coords == "max":
            scale = max(height, width)
            coords_h = torch.arange(0.5, height, device=device) / scale
            coords_w = torch.arange(0.5, width, device=device) / scale
        elif self.normalize_coords == "min":
            scale = min(height, width)
            coords_h = torch.arange(0.5, height, device=device) / scale
            coords_w = torch.arange(0.5, width, device=device) / scale
        else:
            raise ValueError(f"unknown RoPE coordinate normalization: {self.normalize_coords}")

        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        if self.training and self.rescale_coords is not None:
            limit = math.log(self.rescale_coords)
            coords *= torch.empty((), device=device).uniform_(-limit, limit).exp()

        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope_apply(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> torch.Tensor:
    return x * cos + _rope_rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qkv_bias and mask_k_bias:
            bias_mask = torch.ones(dim * 3)
            bias_mask[dim : 2 * dim] = 0
            self.register_buffer("bias_mask", bias_mask, persistent=True)
        else:
            self.bias_mask = None
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, n_tokens, dim = x.shape
        if self.qkv.bias is not None and self.bias_mask is not None:
            qkv = F.linear(x, self.qkv.weight, self.qkv.bias * self.bias_mask)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size, n_tokens, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (token.transpose(1, 2) for token in (q, k, v))

        if rope is not None:
            sin, cos = rope
            prefix = n_tokens - sin.shape[-2]
            if prefix < 0:
                raise ValueError("RoPE has more positions than the token sequence")
            q_patch = _rope_apply(q[:, :, prefix:].float(), sin, cos).to(q.dtype)
            k_patch = _rope_apply(k[:, :, prefix:].float(), sin, cos).to(k.dtype)
            q = torch.cat((q[:, :, :prefix], q_patch), dim=-2)
            k = torch.cat((k[:, :, :prefix], k_patch), dim=-2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, n_tokens, dim)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        layerscale_init: float = 1e-5,
        norm_eps: float = 1e-5,
        mask_k_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, layerscale_init)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, layerscale_init)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope=rope)))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class VisionTransformer(nn.Module):
    """ViT with CLS, storage/register, mask tokens, and axial RoPE."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        n_register_tokens: int = 4,
        use_ibot_mask: bool = True,
        use_rope: bool = True,
        rope_base: float = 100.0,
        rope_rescale_coords: float | None = 2.0,
        layerscale_init: float = 1e-5,
        norm_eps: float = 1e-5,
        untie_global_local_cls_norm: bool = True,
        mask_k_bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_register_tokens = n_register_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(
            torch.empty(1, n_register_tokens, embed_dim)
        )
        self.mask_token = (
            nn.Parameter(torch.empty(1, embed_dim)) if use_ibot_mask else None
        )
        self.rope_embed = (
            RopePositionEmbedding(
                embed_dim,
                num_heads,
                base=rope_base,
                normalize_coords="separate",
                rescale_coords=rope_rescale_coords,
            )
            if use_rope
            else None
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    drop_path_rate,
                    layerscale_init,
                    norm_eps,
                    mask_k_bias,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.local_cls_norm = (
            nn.LayerNorm(embed_dim, eps=norm_eps)
            if untie_global_local_cls_norm
            else None
        )
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.register_tokens, std=0.02)
        if self.mask_token is not None:
            nn.init.zeros_(self.mask_token)
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()

    def _prepare_tokens(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int, int]:
        batch_size, _, height, width = images.shape
        patches = self.patch_embed(images)
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        if mask is not None:
            if self.mask_token is None:
                raise ValueError("mask provided to a backbone without a mask token")
            patches = torch.where(
                mask.unsqueeze(-1),
                self.mask_token.to(patches.dtype).expand_as(patches),
                patches,
            )

        cls = self.cls_token.expand(batch_size, -1, -1)
        registers = self.register_tokens.expand(batch_size, -1, -1)
        return torch.cat((cls, registers, patches), dim=1), patch_h, patch_w

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        is_global: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, patch_h, patch_w = self._prepare_tokens(x, mask)
        for block in self.blocks:
            rope = (
                self.rope_embed(patch_h, patch_w)
                if self.rope_embed is not None
                else None
            )
            x = block(x, rope)

        prefix = 1 + self.n_register_tokens
        if not is_global and self.local_cls_norm is not None and self.training:
            cls = self.local_cls_norm(x[:, 0])
        else:
            cls = self.norm(x[:, 0])
        patches = self.norm(x[:, prefix:])
        return cls, patches


def vit_small_patch16(n_register_tokens: int = 4, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        n_register_tokens=n_register_tokens,
        **kwargs,
    )


def vit_base_patch16(n_register_tokens: int = 4, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        n_register_tokens=n_register_tokens,
        **kwargs,
    )
