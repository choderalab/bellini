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

class ActionableDevice(abc.ABC):
    """ Base class for object that can manipulate experiment objects """

    @abc.abstractmethod
    def apply(self, *args, **kwargs):
        """
        Manipulate the provided experimental objects and return the new objects
        """
        raise NotImplementedError

# =============================================================================
# SUBMODULE CLASS
# =============================================================================

class LiquidTransfer(ActionableDevice):
    """ Transfer an amount of liquid from one container to another with Gaussian error """
    def __init__(self, name, var):
        """
        TODO: allow variance to be drawn from a prior
        """

        assert var.units.dimensionality == VOLUME_UNIT.dimensionality

        if isinstance(var, Quantity):
            self.var = var
        else:
            raise ValueError("var must be either a Quantity")

        self.name = name
        self.dispense_count = 0

    def apply(self, source, sink, volume):
        # independent noise for each array element (TODO: is this valid?)
        if isinstance(source.volume.magnitude, np.ndarray):
            volume = volume * np.ones_like(source.volume.magnitude)
        elif isinstance(source.volume.magnitude, jnp.ndarray):
            volume = volume * jnp.ones_like(source.volume.magnitude)

        # compute drawn volume
        drawn_volume = Normal(volume, self.var)
        drawn_volume.name = f"{self.name}_{drawn_volume.name}_{self.dispense_count}"
        self.dispense_count += 1

        # aliquot and create new containers
        aliquot, new_source = source.retrieve_aliquot(drawn_volume)
        new_sink = sink.receive_aliquot(aliquot)

        return new_source, new_sink
