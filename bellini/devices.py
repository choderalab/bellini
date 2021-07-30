# =============================================================================
# IMPORTS
# =============================================================================

import abc
import pint
import numpy as np
import jax.numpy as jnp
import bellini
from bellini.distributions import Distribution, Normal, gen_lognorm, TruncatedNormal
from bellini.quantity import Quantity
from bellini.containers import Container
from bellini.units import VOLUME_UNIT
from bellini.reference import Reference as Ref

# =============================================================================
# BASE CLASS
# =============================================================================

class Device(abc.ABC):
    """ Base class for devices (objects that don't change over a procedure) """

class ActionableDevice(Device):
    """ Base class for object that can manipulate experiment objects """

    @abc.abstractmethod
    def apply(self, *args, **kwargs):
        """
        Manipulate the provided experimental objects and return the new objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_state(self, exp_state, *args, **kwargs):
        """
        Manipulate the provided experimental state and return the new modified state,
        as well as a belief graph relating the old experimetal state to the new one.
        """
        raise NotImplementedError

class MeasurementDevice(Device):
    """ Base function for measurement instruments """

    @abc.abstractmethod
    def readout(self, *args, **kwargs):
        """
        Return measurement(s) of the given objects (generally with some error)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def readout_state(self, exp_state, *args, **kwargs):
        """
        Return measurement(s) of the experimental state (generally with some error)
        given reference to the particular objects to measure
        """
        raise NotImplementedError

# =============================================================================
# SUBMODULE CLASS
# =============================================================================

class LiquidTransfer(ActionableDevice):
    """ Transfer an amount of liquid from one container to another with Gaussian error """
    _SUPPORTED_DISTS = ["Normal", "LogNormal", "TruncatedNormal", None]
    def __init__(self, name, var, noise_model="Normal"):
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
        self.noise_model = noise_model
        assert self.noise_model in LiquidTransfer._SUPPORTED_DISTS

    def _noisy_volume(self, volume):
        if self.noise_model == "Normal":
            return Normal(volume, self.var)
        elif self.noise_model == "LogNormal":
            return gen_lognorm(volume, self.var)
        elif self.noise_model == "TruncatedNormal":
            return TruncatedNormal(Quantity(0, self.var.units), volume, self.var)
        elif self.noise_model is None:
            return volume
        else:
            raise ValueError(f"noise model param of {self} is not valid")

    def apply(self, source, sink, volume):
        # independent noise for each array element (TODO: is this valid?)
        if isinstance(source.volume.magnitude, np.ndarray):
            volume = volume * np.ones_like(source.volume.magnitude)
        elif isinstance(source.volume.magnitude, jnp.ndarray):
            volume = volume * jnp.ones_like(source.volume.magnitude)

        # compute drawn volume
        drawn_volume = self._noisy_volume(volume)
        drawn_volume.name = f"{self.name}_{drawn_volume.name}_{self.dispense_count}"
        self.dispense_count += 1

        # aliquot and create new containers
        aliquot, new_source = source.retrieve_aliquot(drawn_volume)
        new_sink = sink.receive_aliquot(aliquot)

        return new_source, new_sink

    def apply_state(self, experiment_state, source_ref, sink_ref, volume):
        # retrieve source and sink containers
        if isinstance(source_ref, Ref):
            source_outer = experiment_state[source_ref.name]
            source = source_ref.retrieve_index(source_outer)
        elif isinstance(source_ref, str):
            source = experiment_state[source_ref]
        else:
            raise ValueError(f"source_ref must be Reference or str, but is {source_ref}")

        if isinstance(sink_ref, Ref):
            sink_outer = experiment_state[sink_ref.name]
            sink = sink_ref.retrieve_index(sink_outer)
        elif isinstance(sink_ref, str):
            sink = experiment_state[sink_ref]
        else:
            raise ValueError(f"sink_ref must be Reference or str, but is {sink_ref}")
        assert isinstance(source, Container)
        assert isinstance(sink, Container)

        # get new source and sink
        new_source, new_sink = self.apply(source, sink, volume)

        # aliquot and create new experiment state
        new_experiment_state = experiment_state.copy()
        if isinstance(source_ref, Ref):
            source_ref.set_index(new_experiment_state[source_ref.name], new_source, copy=False)
        else:
            new_experiment_state[source_ref] = new_source
        if isinstance(sink_ref, Ref):
            sink_ref.set_index(new_experiment_state[sink_ref.name], new_sink, copy=False)
        else:
            new_experiment_state[sink_ref] = new_sink

        # generate belief graph
        belief_graph = {
            new_source: (source,),
            new_sink: (source, sink),
        }

        return new_experiment_state, belief_graph

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
        self.units = self.var.units

    def readout(self, container, value, key=None):
        prenoise_value = container.observe(value, key=key)
        assert prenoise_value.dimensionality == self.units.dimensionality
        measurement = Normal(prenoise_value, self.var)
        measurement.name = f"{self.name}_{container}_{value}_{self.measure_count}"
        self.measure_count += 1
        measurement.observed = True
        return measurement

    def readout_state(self, experiment_state, container_ref, value, key=None):
        if isinstance(container_ref, Ref):
            container_outer = experiment_state[container_ref.name]
            container = container_ref.retrieve_index(container_outer)
        elif isinstance(container_ref, str):
            container = experiment_state[container_ref]
        else:
            raise ValueError(f"container_ref must be Reference or str, but is {container_ref}")
        measurement = self.readout(container, value, key)
        return {(value, key): measurement} if key else {value: measurement}
