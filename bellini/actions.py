# =============================================================================
# IMPORTS
# =============================================================================

import abc
import pint
import numpy as np
import bellini
from bellini.distributions import Distribution, Normal
from bellini.quantity import Quantity
from bellini.units import *

# =============================================================================
# BASE CLASS
# =============================================================================

class Actionable(abc.ABC):
    """ Base class for common object that can apply an action """
    def apply(self, parent_node, child_node, *args):
        raise NotImplementedError

# =============================================================================
# SUBMODULE CLASS
# =============================================================================

class Dispenser(Actionable):
    """ Dispense an amount of liquid from one container to another with Gaussian error """
    def __init__(self, name, var):
        """
        TODO: allow variance to be drawn from a prior
        """

        #assert var.units == VOLUME_UNIT

        if isinstance(var, Quantity):
            self.var = var
        else:
            raise ValueError("var must be either a Quantity")

        self.name = name
        self.dispense_count = 0

    def apply(self, parent_node, child_node, volume, parent_final_name=None, child_final_name=None):
        drawn_volume = Normal(volume, self.var)
        drawn_volume.name = f"vol_transfer_{drawn_volume.name}_{self.dispense_count}"
        self.dispense_count += 1
        aliquot = parent_node.retrieve_aliquot(drawn_volume)
        child_node.recieve_aliquot(aliquot)



class Measurer(Actionable):
    """ Measure a property of one container with Gaussian error """
    def __init__(self, name, var):
        """
        TODO: allow variance to be drawn from a prior, update name for multiple
        measurements
        """

        if isinstance(var, Quantity):
            self.var = var
        else:
            raise ValueError("var must be a Quantity")

        self.name = name
        self.measure_count = 0

    def apply(self, parent_node, value):
        measurement = Normal(parent_node.observe(value), self.var)
        measurement.name = f"measurement_{parent_node}_{value}_{self.measure_count}"
        self.measure_count += 1
        measurement.observed = True
        return measurement
