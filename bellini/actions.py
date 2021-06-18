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
        We allow Quantity as a variance if the error of the dispensing device is known
        Otherwise, var can be a distribution which serves as a prior (must be positive RV)
        """

        #assert var.units == VOLUME_UNIT

        if isinstance(var, Quantity):
            self.var = var
            self.noise_model = Normal(0 * VOLUME_UNIT, var)
        elif isinstance(var, Distribution):
            self.var = var
            self.noise_model = Normal(0 * VOLUME_UNIT, 1  * VOLUME_UNIT) * var
        else:
            raise InvalidArgumentError("var must be either a Quantity or a Distribution")

        self.name = name

    def apply(self, parent_node, child_node, volume):
        drawn_volume = self.noise_model + volume
        aliquot = parent_node.retrieve_aliquot(drawn_volume)
        child_node.recieve_aliquot(aliquot)

class Measurer(Actionable):
    """ Measure a property of one container with Gaussian error """
    def __init__(self, name, var):
        """
        We allow Quantity as a variance if the error of the dispensing device is known
        Otherwise, var can be a distribution which serves as a prior (must be positive RV)
        """

        #assert var.units == VOLUME_UNIT

        if isinstance(var, Quantity):
            self.var = var
            self.noise_model = Normal(0 * var.units, var)
        elif isinstance(var, Distribution):
            self.var = var
            self.noise_model = Normal(0 * var.units, 1  * var.units) * var
        else:
            raise InvalidArgumentError("var must be either a Quantity or a Distribution")

        self.name = name

    def apply(self, parent_node, value):
        measurement = self.noise_model + parent_node.observe(value)
        measurement.observed = True
        return measurement
