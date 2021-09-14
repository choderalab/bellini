""" A set of utility functions that are useful when interacting with a mix of
Quantities, Distributions, and other numerical primitives """

import jax.numpy as jnp
import numpy as np
import torch
import bellini
from bellini.units import ureg, to_internal_units

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
    """ Checks if the shape of `a` and `b` is the same """
    if not hasattr(a, "shape") or not hasattr(b, "shape"):
        # automatically true if one of them is a scalar
        return True
    if a.shape != () and b.shape != ():
        return a.shape == b.shape
    return True

def check_broadcastable(*args):
    """ Check if the provided args are broadcastable """
    try:
        shape_args = [np.empty(arg.shape) for arg in args]
        np.broadcast(*shape_args)
        return True
    except ValueError:
        return False

def flatten(args, keep_keys=False):
    """ Flatten a nested set of lists, tuples, and dicts """
    ret = []
    if isinstance(args, dict):
        for key, value in args.items():
            if keep_keys:
                ret += flatten(key)
            ret += flatten(value)
    elif isinstance(args, (list, tuple)):
        for arg in args:
            ret += flatten(arg)
    else:
        ret.append(args)
    return ret

def _to_x_constructor(fn):
    """ Apply a function `fn` to all elements of lists and tuples, as well
    as values in dicts, in a nested set of lists, tuples, and dicts """
    def _to_x(args):
        if isinstance(args, dict):
            return {
                key: _to_x(value)
                for key, value in args.items()
            }
        elif isinstance(args, list):
            return [_to_x(arg) for arg in args]
        elif isinstance(args, tuple):
            return tuple([_to_x(arg) for arg in args])
        else:
            return fn(args)
    return _to_x

def _to_quantity(arg):
    """ Convert `arg` to a deterministic Quantity """
    if isinstance(arg, bellini.Distribution):
        return bellini.Quantity(arg.magnitude, arg.units)
    elif isinstance(arg, bellini.Quantity):
        return arg
    elif is_arr(arg) or is_scalar(arg):
        return bellini.Quantity(arg)
    else:
        raise ValueError(f"unable to convert {arg} to Quantity")

args_to_quantity = _to_x_constructor(_to_quantity)
""" A function that converts all values in args to Quantities """
