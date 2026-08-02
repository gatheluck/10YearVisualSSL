"""
iBOT: Image BERT Pre-Training with Online Tokenizer.

Reference: Zhou et al., arXiv:2111.07832
           https://github.com/bytedance/ibot

Architecture:
  Student ViT  (with learnable [MASK] token)
  Teacher ViT  (EMA of student — serves as the online tokenizer)
  Shared DINOHead applied to both [CLS] and patch tokens
  iBOT loss = CLS self-distillation loss + patch MIM loss

The teacher is NOT updated by gradient; it follows an EMA schedule.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ── DINO Projection Head ─────────────────────────────────────────────────────

class DINOHead(nn.Module):
    """
    3-layer MLP projection head.
      in_dim -> hidden_dim -> hidden_dim -> bottleneck_dim [L2-norm] -> out_dim
    The final linear layer uses weight normalization (no bias), same as DINO.
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        nlayers=3,
        norm_last_layer=True,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (nlayers - 1) + [bottleneck_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

        # Weight-norm final layer (no bias), with optional frozen norm.
        # Use the new parametrizations API (PyTorch >= 1.11) to avoid deepcopy issues.
        linear = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.parametrizations.weight_norm(linear)
        nn.init.constant_(self.last_layer.parametrizations.weight.original0, 1)  # weight_g = 1
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        self.apply(_init_linear)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def _init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ── iBOT Loss ────────────────────────────────────────────────────────────────

class iBOTLoss(nn.Module):
    """
    Combined CLS self-distillation loss + patch MIM loss.

    Both use the same cross-entropy formulation (DINO-style):
      L = -sum_t [softmax(teacher_out / t_t) * log_softmax(student_out / t_s)]

    Teacher outputs are centered before computing softmax.
    Two separate centers are maintained: one for CLS tokens, one for patch tokens.

    Args:
        out_dim          : Number of visual token prototypes K (e.g., 8192).
        patch_out_dim    : Can differ from out_dim; set equal to out_dim.
        student_temp     : Student softmax temperature (default 0.1).
        teacher_temp     : Teacher softmax temperature after warmup (default 0.07).
        teacher_temp_warmup: Teacher temp at start of warmup (default 0.04).
        teacher_temp_warmup_epochs: Epochs to ramp up teacher temp (default 30).
        center_momentum  : EMA for centering (default 0.9).
        lambda_token     : Weight on patch MIM loss (default 1.0).
        n_global_crops   : Number of global views (default 2).
        n_local_crops    : Number of local views (default 10).
    """

    def __init__(
        self,
        out_dim,
        patch_out_dim,
        student_temp=0.1,
        teacher_temp=0.07,
        teacher_patch_temp=None,
        teacher_temp_warmup=0.04,
        teacher_patch_temp_warmup=None,
        teacher_temp_warmup_epochs=30,
        center_momentum=0.9,
        center_momentum_patch=None,
        lambda_token=1.0,
        n_global_crops=2,
        n_local_crops=10,
    ):
        super().__init__()
        self.student_temp   = student_temp
        self.teacher_temp   = teacher_temp
        self.teacher_patch_temp = teacher_patch_temp if teacher_patch_temp is not None else teacher_temp
        self.teacher_temp_warmup        = teacher_temp_warmup
        self.teacher_patch_temp_warmup = (
            teacher_patch_temp_warmup if teacher_patch_temp_warmup is not None else teacher_temp_warmup
        )
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.center_momentum = center_momentum
        self.center_momentum_patch = center_momentum_patch if center_momentum_patch is not None else center_momentum
        self.lambda_token   = lambda_token
        self.n_global_crops = n_global_crops
        self.n_local_crops  = n_local_crops
        self.n_crops        = n_global_crops + n_local_crops

        # Running centers (updated per batch via EMA)
        self.register_buffer("center_cls",   torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, patch_out_dim))

    def get_teacher_temp(self, epoch):
        if self.teacher_temp_warmup_epochs == 0:
            return self.teacher_temp
        if epoch < self.teacher_temp_warmup_epochs:
            if self.teacher_temp_warmup_epochs == 1:
                return self.teacher_temp
            return self.teacher_temp_warmup + (self.teacher_temp - self.teacher_temp_warmup) * (
                epoch / (self.teacher_temp_warmup_epochs - 1)
            )
        return self.teacher_temp

    def get_teacher_patch_temp(self, epoch):
        if self.teacher_temp_warmup_epochs == 0:
            return self.teacher_patch_temp
        if epoch < self.teacher_temp_warmup_epochs:
            if self.teacher_temp_warmup_epochs == 1:
                return self.teacher_patch_temp
            return self.teacher_patch_temp_warmup + (
                self.teacher_patch_temp - self.teacher_patch_temp_warmup
            ) * (epoch / (self.teacher_temp_warmup_epochs - 1))
        return self.teacher_patch_temp

    def forward(
        self,
        student_cls_list,
        teacher_cls_list,
        student_patch_list,
        teacher_patch_list,
        mask_list,
        epoch,
    ):
        """
        Args:
            student_cls_list   : list of [B, K] student CLS logits, length = n_crops
            teacher_cls_list   : list of [B, K] teacher CLS logits, length = n_global_crops
            student_patch_list : list of [B, N, K] student patch logits, length = n_global_crops
            teacher_patch_list : list of [B, N, K] teacher patch logits, length = n_global_crops
            mask_list          : list of [B, N] boolean masks, length = n_global_crops
                                 True = this patch was masked in student input
            epoch              : current epoch (for teacher temp schedule)
        Returns:
            total_loss, cls_loss, patch_loss
        """
        teacher_temp = self.get_teacher_temp(epoch)
        teacher_patch_temp = self.get_teacher_patch_temp(epoch)

        # ── Compute teacher probabilities (with centering) ──────────────────

        # CLS: center subtraction + softmax
        teacher_cls_probs = [
            F.softmax((t - self.center_cls) / teacher_temp, dim=-1).detach()
            for t in teacher_cls_list
        ]
        # Patch: center subtraction + softmax
        teacher_patch_probs = [
            F.softmax((t - self.center_patch) / teacher_patch_temp, dim=-1).detach()
            for t in teacher_patch_list
        ]

        # ── CLS self-distillation loss ───────────────────────────────────────
        total_cls_loss = 0.0
        n_cls_pairs    = 0

        for s_idx in range(self.n_crops):
            s_log_probs = F.log_softmax(student_cls_list[s_idx] / self.student_temp, dim=-1)
            for t_idx, t_probs in enumerate(teacher_cls_probs):
                if s_idx == t_idx:
                    continue  # skip same-view pairs
                total_cls_loss += -(t_probs * s_log_probs).sum(dim=-1).mean()
                n_cls_pairs += 1

        cls_loss = total_cls_loss / max(n_cls_pairs, 1)

        # ── Patch MIM loss ───────────────────────────────────────────────────
        total_patch_loss = 0.0
        n_patch_pairs    = 0

        for g_idx in range(self.n_global_crops):
            mask = mask_list[g_idx]  # [B, N]
            if mask.sum() == 0:
                continue
            t_patch_probs = teacher_patch_probs[g_idx]  # [B, N, K]
            s_patch_logits = student_patch_list[g_idx]  # [B, N, K]

            s_log_probs = F.log_softmax(s_patch_logits / self.student_temp, dim=-1)  # [B, N, K]
            patch_loss = -(t_patch_probs * s_log_probs).sum(dim=-1)  # [B, N]
            # Official iBOT normalizes masked-patch CE per sample, then averages
            # across the batch. This keeps zero-mask samples in the batch mean.
            patch_loss = (patch_loss * mask.float()).sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1.0)
            patch_loss = patch_loss.mean()

            total_patch_loss += patch_loss
            n_patch_pairs += 1

        patch_loss = total_patch_loss / max(n_patch_pairs, 1) if n_patch_pairs > 0 else \
                     student_patch_list[0].sum() * 0.0

        total_loss = cls_loss + self.lambda_token * patch_loss

        # ── Update centers (EMA over teacher outputs) ───────────────────────
        with torch.no_grad():
            # CLS center: average over all teacher global crop outputs
            cls_cat = torch.cat(teacher_cls_list, dim=0)  # [n_global*B, K]
            self._update_center(cls_cat, self.center_cls, self.center_momentum)

            # Patch center: average over all non-masked patch tokens
            # Flatten all global view patch outputs
            patch_cat = torch.cat(teacher_patch_list, dim=0).reshape(-1, teacher_patch_list[0].shape[-1])
            self._update_center(patch_cat, self.center_patch, self.center_momentum_patch)

        return total_loss, cls_loss, patch_loss

    @torch.no_grad()
    def _update_center(self, batch_out, center, momentum):
        """EMA update of centering buffer, with all-reduce for DDP."""
        batch_mean = batch_out.mean(dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_mean)
            batch_mean /= dist.get_world_size()
        center.copy_(center * momentum + batch_mean * (1 - momentum))


# ── EMA Teacher Update ───────────────────────────────────────────────────────

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    """EMA update: teacher_param = m * teacher_param + (1-m) * student_param.
    Uses named parameter matching to handle the case where student has extra
    parameters (e.g., mask_token) that teacher does not.
    """
    teacher_sd = {n: p for n, p in teacher.named_parameters()}
    for name, ps in student.named_parameters():
        if name in teacher_sd and teacher_sd[name].shape == ps.shape:
            teacher_sd[name].data.mul_(momentum).add_((1 - momentum) * ps.detach().data)


def cosine_teacher_momentum(base_mom, final_mom, epoch, total_epochs):
    """Cosine schedule for teacher EMA momentum: base_mom -> final_mom."""
    return final_mom - (final_mom - base_mom) * (
        math.cos(math.pi * epoch / total_epochs) + 1
    ) / 2


# ── Full iBOT Model ──────────────────────────────────────────────────────────

class iBOT(nn.Module):
    """
    iBOT model wrapping student + teacher ViTs and a shared DINOHead.

    The teacher is initialized as a copy of the student and updated via EMA.
    No gradients flow through the teacher.
    """

    def __init__(self, student_vit, teacher_vit, head):
        super().__init__()
        self.student = student_vit
        self.teacher = teacher_vit
        self.head    = head

        # Teacher head (EMA of student head)
        import copy
        self.teacher_head = copy.deepcopy(head)

        # Teacher gets no gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Initialise teacher = student
        self._init_teacher()

    def _init_teacher(self):
        # Copy only matching named parameters (student may have mask_token, teacher does not)
        student_sd = self.student.state_dict()
        teacher_sd = self.teacher.state_dict()
        for k in teacher_sd:
            if k in student_sd and student_sd[k].shape == teacher_sd[k].shape:
                teacher_sd[k].copy_(student_sd[k])
        self.teacher.load_state_dict(teacher_sd)
        # Head weights are identical shapes, safe to copy directly
        teacher_head_sd = self.teacher_head.state_dict()
        head_sd         = self.head.state_dict()
        for k in teacher_head_sd:
            if k in head_sd and head_sd[k].shape == teacher_head_sd[k].shape:
                teacher_head_sd[k].copy_(head_sd[k])
        self.teacher_head.load_state_dict(teacher_head_sd)

    @torch.no_grad()
    def update_teacher(self, momentum):
        """Call once per iteration after the optimizer step.
        Updates both backbone and head via EMA.
        """
        update_teacher(self.student,  self.teacher,      momentum)
        update_teacher(self.head,     self.teacher_head, momentum)

    def get_encoder(self):
        """Return the student ViT backbone (without head) for linear probing."""
        return self.student

    def forward(self, crops, masks, n_global=None):
        """DDP-visible forward for student and teacher outputs."""
        if n_global is None:
            n_global = len(masks)
        student_cls_list, student_patch_list = self.forward_student(crops, masks)
        teacher_cls_list, teacher_patch_list = self.forward_teacher(crops[:n_global])
        return student_cls_list, student_patch_list, teacher_cls_list, teacher_patch_list

    def forward_student(self, crops, masks):
        """
        Forward pass through student on all crops.

        Args:
            crops : list of tensors [B, C, H, W]; first n_global are global views.
            masks : list of [B, N] boolean masks for global views; None for local views.

        Returns:
            cls_list  : list of [B, K] student CLS logits, length=n_crops
            patch_list: list of [B, N, K] student patch logits, length=n_global
        """
        cls_list   = []
        patch_list = []
        n_global   = len(masks)

        for i, img in enumerate(crops):
            mask = masks[i] if i < n_global else None
            cls_tok, patch_tok = self.student(img, mask=mask)
            cls_list.append(self.head(cls_tok))
            if i < n_global:
                # Apply head to each patch token independently
                B, N, D = patch_tok.shape
                patch_out = self.head(patch_tok.reshape(B * N, D)).reshape(B, N, -1)
                patch_list.append(patch_out)

        return cls_list, patch_list

    @torch.no_grad()
    def forward_teacher(self, global_crops):
        """
        Forward pass through teacher on clean global crops (no masking).

        Returns:
            cls_list  : list of [B, K] teacher CLS logits
            patch_list: list of [B, N, K] teacher patch logits
        """
        cls_list   = []
        patch_list = []
        for img in global_crops:
            cls_tok, patch_tok = self.teacher(img, mask=None)
            cls_list.append(self.teacher_head(cls_tok))
            B, N, D = patch_tok.shape
            patch_out = self.teacher_head(patch_tok.reshape(B * N, D)).reshape(B, N, -1)
            patch_list.append(patch_out)
        return cls_list, patch_list
