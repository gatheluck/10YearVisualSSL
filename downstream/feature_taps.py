"""Explicit intermediate feature collection."""
import math


def even_taps(n_layers):
    if type(n_layers) is not int or n_layers < 4:
        raise ValueError('four taps require at least four layers')
    return [math.ceil(k * n_layers / 4) - 1 for k in (1, 2, 3, 4)]


def collect_block_taps(blocks, indices, forward):
    indices = list(indices)
    if len(indices) != 4 or any(type(i) is not int or not 0 <= i < len(blocks) for i in indices) or len(set(indices)) != 4:
        raise ValueError('select four distinct valid block indices')
    outputs = {}
    handles = []
    def hook(index):
        def capture(module, args, output):
            outputs[index] = output[0] if isinstance(output, (tuple, list)) else output
        return capture
    try:
        for index in indices:
            handles.append(blocks[index].register_forward_hook(hook(index)))
        forward()
        missing = [i for i in indices if i not in outputs]
        if missing:
            raise RuntimeError(f'tap hooks did not fire: {missing}')
        return [outputs[i] for i in indices]
    finally:
        for handle in handles:
            handle.remove()


def tokens_to_grid(tokens, height, width, *, prefix=0):
    if tokens.ndim != 3 or type(prefix) is not int or prefix < 0 or type(height) is not int or type(width) is not int or height <= 0 or width <= 0 or tokens.shape[1] - prefix != height * width:
        raise ValueError('token count and prefix must match the spatial grid')
    return tokens[:, prefix:].transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)
