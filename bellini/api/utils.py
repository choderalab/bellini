import jax.numpy as jnp
import numpy as np
import torch
import bellini
from bellini.units import *

def isarr(arr):
    if isinstance(arr, bellini.Quantity):
        arr = arr.magnitude
    return isinstance(arr, np.ndarray) or isinstance(arr, jnp.ndarray) or isinstance(arr, torch.Tensor)

def mask(arr, idxs, invert=False):
    if invert:
        select = np.ones_like(arr)
        select[idxs] = 0
    else:
        select = np.ones_like(arr)
        select[idxs] = 1
    return bellini.Quantity(select, ureg.dimensionless)
