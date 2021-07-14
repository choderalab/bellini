# =============================================================================
# IMPORTS
# =============================================================================

import abc
import pint
import numpy as np
import jax.numpy as jnp
import bellini
from bellini.distributions import Distribution, Normal, gen_lognorm
from bellini.quantity import Quantity
from bellini.containers import Container
from bellini.units import VOLUME_UNIT

# =============================================================================
# BASE CLASS
# =============================================================================

class MeasurementDevice(abc.ABC):
    """ Base function for measurement instruments """

    @abc.abstractmethod
    def readout(self, *args, **kwargs):
        """
        Return measurement(s) of the given objects (generally with some error)
        """
        raise NotImplementedError

# =============================================================================
# SUBMODULE CLASS
# =============================================================================

class Measurer(MeasurementDevice):
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

    def readout(self, container, value, key=None):
        measurement = Normal(container.observe(value, key=key), self.var)
        measurement.name = f"{self.name}_{container}_{value}_{self.measure_count}"
        self.measure_count += 1
        measurement.observed = True
        return measurement
