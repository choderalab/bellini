# =============================================================================
# IMPORTS
# =============================================================================
import pint
import numpy as np

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Quantity(pint.quantity.Quantity):
    """ A class that describes physical quantity, which contains
    numeric value and units.

    """

    @staticmethod
    def _convert_to_numpy(x):
        if isinstance(x, float):
            return x
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    def __new__(self, value, unit, name=None):
        value = self._convert_to_numpy(value)
        if name is None:
            name = repr(self)
        self.name = name
        return super(Quantity, self).__new__(self, value, unit)
