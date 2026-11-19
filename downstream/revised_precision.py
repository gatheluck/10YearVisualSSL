"""Scoped numerical backend policy for the inspected revised trainers."""
from contextlib import contextmanager
from typing import Any, cast

import torch


def _operator_precision():
    return hasattr(torch.backends.cuda.matmul, 'fp32_precision')


def _fp32_policy():
    backend = cast(Any, torch.backends)  # Version-dependent dynamic backend properties.
    if _operator_precision():
        # Do not mix new operator policies with legacy TF32 getters: PyTorch
        # rejects that combination for valid mixed conv/RNN policies.
        return {'api': 'operator', 'global': backend.fp32_precision,
                'cuda_matmul': torch.backends.cuda.matmul.fp32_precision,
                'cudnn': torch.backends.cudnn.fp32_precision,
                'cudnn_conv': backend.cudnn.conv.fp32_precision,
                'cudnn_rnn': backend.cudnn.rnn.fp32_precision}
    return {'api': 'legacy', 'float32_matmul': torch.get_float32_matmul_precision(),
            'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_tf32': torch.backends.cudnn.allow_tf32}


def runtime_precision():
    """Observe numerical switches without initializing CUDA or changing them.

    These settings identify a runtime policy, not a hardware or numeric-parity
    certificate. Record inherited switches rather than inventing source values.
    """
    return {
        'fp32': _fp32_policy(),
        'cudnn_enabled': torch.backends.cudnn.enabled,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cuda_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'cuda_bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'sdpa_flash': torch.backends.cuda.flash_sdp_enabled(),
        'sdpa_memory_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'sdpa_math': torch.backends.cuda.math_sdp_enabled(),
    }


@contextmanager
def precision_policy(device, *, matmul_tf32, cudnn_tf32=None):
    """Scope an explicit CUDA policy without altering unrelated settings.

    New PyTorch uses operator-specific settings, without legacy getters/setters.
    Older PyTorch preserves the full matmul mode, including ``medium``: a
    boolean alone cannot restore it. CPU does not change any backend policy.
    ``cudnn_tf32=None`` leaves convolution/RNN policy unchanged.
    """
    if type(matmul_tf32) is not bool or (cudnn_tf32 is not None and type(cudnn_tf32) is not bool):
        raise ValueError('precision switches must be explicit booleans')
    if device.type != 'cuda':
        yield
        return
    previous = _fp32_policy()
    backend = cast(Any, torch.backends)
    try:
        if previous['api'] == 'operator':
            torch.backends.cuda.matmul.fp32_precision = 'tf32' if matmul_tf32 else 'ieee'
            if cudnn_tf32 is not None:
                backend.cudnn.conv.fp32_precision = 'tf32' if cudnn_tf32 else 'ieee'
                backend.cudnn.rnn.fp32_precision = 'tf32' if cudnn_tf32 else 'ieee'
        else:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            if cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
        yield
    finally:
        if previous['api'] == 'operator':
            torch.backends.cuda.matmul.fp32_precision = previous['cuda_matmul']
            if cudnn_tf32 is not None:
                backend.cudnn.conv.fp32_precision = previous['cudnn_conv']
                backend.cudnn.rnn.fp32_precision = previous['cudnn_rnn']
        else:
            torch.set_float32_matmul_precision(previous['float32_matmul'])
            if cudnn_tf32 is not None:
                torch.backends.cudnn.allow_tf32 = previous['cudnn_tf32']


def source_precision(device):
    """The revised LoRA/DPT source enables both CUDA TF32 switches."""
    return precision_policy(device, matmul_tf32=True, cudnn_tf32=True)
