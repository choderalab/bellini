""" A set of utility functions that are useful when interacting with a mix of
Quantities, Distributions, and other numerical primitives """

import jax.numpy as jnp
import numpy as np
import torch
import bellini
from bellini.units import *

def is_scalar(num):
    """ Returns if `num` is a scalar i.e. float, int, or `np.generic`

    Parameters
    ----------
    num : object

    Returns
    -------
    bool
        Whether or not `num` is a scalar
    """
    return isinstance(num, (float, int, np.generic))

def is_arr(arr):
    """ Returns if `num` is either an array or a `Quantity` holding an array

    Parameters
    ----------
    arr : object

    Returns
    -------
    bool
        Whether or not `arr` is either an array or a `Quantity` holding an array
    """
    if isinstance(arr, bellini.Quantity):
        arr = arr.magnitude
    return isinstance(arr, np.ndarray) or isinstance(arr, jnp.ndarray) or isinstance(arr, torch.Tensor)

def mask(arr, idxs, invert=False):
    """ Returns a mask for `arr` at positions `idxs`

    Parameters
    ----------
    arr : array-like
        The object to be masked
    idxs : array-like
        The indices to be masked
    invert : bool, default=False
        If `True`, positions specified by `idxs` will be 0, otherwise 1

    Returns
    -------
    mask : `Quantity` (dimensionless)
        Mask for `arr`
    """
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
