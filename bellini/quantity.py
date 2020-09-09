# =============================================================================
# IMPORTS
# =============================================================================
import pint
import numpy as np
import torch

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Quantity(pint.quantity.Quantity):
    """ A class that describes physical quantity, which contains
    numeric value and units.

    """

    mutable = False

    @staticmethod
    def _convert_to_numpy(x):
        if isinstance(x, float):
            return x
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            # TODO:
            # do not require torch import ahead of time
            return x.numpy()

    def __new__(self, value, unit):
        value = self._convert_to_numpy(value)
        return super(Quantity, self).__new__(self, value, unit)
