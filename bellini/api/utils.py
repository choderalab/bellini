import jax.numpy as jnp
import numpy as np
import torch
import bellini
from bellini.units import *

def is_scalar(num):
    return isinstance(num, float) or isinstance(num, int)

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

def check_shape(a, b):
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        return True
    if a.shape != () and b.shape != ():
        return a.shape == b.shape
    return True
