# =============================================================================
# IMPORTS
# =============================================================================
import abc
import pint
import numpy as np
import torch

# =============================================================================
# BASE CLASS
# =============================================================================
class Group(abc.ABC):
    """ Base class for groups that hold quantities and children. """

    def __init__(self, name, **quantities):
        self.name = name

        # sanity check quantities
        assert all(
            isinstance(quantity, be.quantity.Quantity)
            for quantity in quantities), "quantities have to be `Quantity`"

        self.quantities = quantities

    def __getattr__(self, name):
        if name in self.quantities:
            return self.quantities[name]

        else:
            AttributeError(
                "%s has no attribute %s" % (
                    self.name,
                    name,
                ))
