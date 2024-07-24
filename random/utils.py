import torch
import functools
import operator
from typing import Tuple

Shape = Tuple[int]

def volume(shape: Shape) -> int:
    return functools.reduce(operator.mul, shape, 1)

def philox_cuda_seed_offset(increment, device=None):
    device = device or torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state_copy = gen.get_state()
    view = state_copy.view(torch.int64)
    c0 = view[0]
    c1 = view[1]
    seed, offset = int(c0), int(c1)
    increment = (increment + 3) // 4 * 4
    c1 += increment
    # get_state returns a new tensor, so it needs set_state to update the actual generator state.
    gen.set_state(state_copy)
    return seed, offset
