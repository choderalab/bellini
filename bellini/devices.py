"""
Module containing various experimental devices, which are used to manipuate
either experimental objects or experimental states
"""
# =============================================================================
# IMPORTS
# =============================================================================

import abc
import pint
import numpy as np
import jax.numpy as jnp
from jax import lax
import bellini
import bellini.api.functional as F
from bellini.distributions import Distribution, Normal, gen_lognorm, TruncatedNormal
from bellini.quantity import Quantity
from bellini.containers import Container
from bellini.units import VOLUME_UNIT
from bellini.reference import Reference as Ref
import warnings

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
    """ Transfer an amount of liquid from one container to another with specified error """
    _SUPPORTED_DISTS = ["Normal", "LogNormal", "TruncatedNormal", None]
    def __init__(self, name, var, noise_model="Normal"):
        """
        Parameters
        ----------
        name: str
            Name of the LiquidTransfer device. Will be used in assigning names
            to each volume transfer sample

        var: Quantity (volume units)
            Error in volume drawn

        noise_model: str, default="Normal"
            Noise model that LiquidTransfer uses. Choices include "Normal",
            "TruncatedNormal", "LogNormal", and `None`.

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
        """ Transfer `volume` from `source` to `sink`

        Arguments
        ---------
        source : Container (not empty)
            Container the aliquot is drawn from

        sink: Container
            Container the aliquot is placed in

        volume: Quantity or Distribution (volume units)
            Amount to transfer

        Returns
        -------
        new_source: Container
            `source` after the aliquot has been removed

        new_sink : Container
            `sink` after the aliquot has been removed
        """
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
        """ Transfer `volume` from `source_ref` in `experimental_state` to
        `sink_ref` in `experimental_state`

        Arguments
        ---------
        experimental_state : dict
            Current experimental state

        source_ref : Reference
            Reference to Container the aliquot is drawn from

        sink_ref : Reference
            Reference to Container the aliquot is placed in

        volume: Quantity or Distribution (volume units)
            Amount to transfer

        Returns
        -------
        new_experiment_state : dict
            Experimental state after volume transfer

        belief_graph : dict
            dict of (current experimental object) -> (all dependent previous
            experimental objects)

        """
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
        Parameters
        ----------
        name: str
            Name of the LiquidTransfer device. Will be used in assigning names
            to each volume transfer sample

        var: Distribution (volume units)
            Error in volume drawn.

        TODO: allow variance to be drawn from a prior
        """

        if isinstance(var, Quantity):
            self.var = var
        else:
            raise ValueError("var must be a Quantity")

        self.name = name
        self.measure_count = 0
        self.units = self.var.units

    def readout(self, container, value, key=None):
        """ Readout `value` from `container`

        Arguments
        ---------
        container : Container (not empty)
            Container `value` is readout from

        value : Reference or str
            What attribute to readout from `container`

        Returns
        -------
        measurement 
            The measured value with some Gaussian noise
        """
        prenoise_value = container.observe(value, key=key)
        assert prenoise_value.dimensionality == self.units.dimensionality
        measurement = Normal(prenoise_value, self.var)
        measurement.name = f"{self.name}_{container}_{value}_{self.measure_count}"
        self.measure_count += 1
        measurement.observed = True
        return measurement

    def readout_state(self, experiment_state, container_ref, value, key=None):
        """ Readout `value` from `container_ref` in `experimental_state`

        Arguments
        ---------
        experimental_state : dict
            Current experimental state

        container_ref : Reference or str
            Reference to Container `value` is readout from

        value : Reference or str
            What attribute to readout from `container`

        Returns
        -------
        measurement_dict : dict
            dict of `value` -> measurement, the measured value with some
            Gaussian noise
        """
        if isinstance(container_ref, Ref):
            container_outer = experiment_state[container_ref.name]
            container = container_ref.retrieve_index(container_outer)
        elif isinstance(container_ref, str):
            container = experiment_state[container_ref]
        else:
            raise ValueError(f"container_ref must be Reference or str, but is {container_ref}")
        measurement = self.readout(container, value, key)
        return {(value, key): measurement} if key else {value: measurement}


class Routine(Device):
    """ [EXPERIMENTAL] Allows efficient repetition of a procedure subroutine
    in numpyro. """
    def __init__(self, objs_to_carry, carry_to_objs, subroutine,
                 output_units, params, measure_var=None):
        if bellini.backend != "numpyro":
            warnings.warn("Routines are numpyro-specific and have not been "
            "tested on different backends")
        self.compress = objs_to_carry
        self.extract = carry_to_objs
        self.subroutine = subroutine
        self.output_units = output_units
        self.params = params
        self.measure_var = measure_var
        if measure_var:
            assert measure_var.dimensionality == output_units.dimensionality

    def perform(self, objs, xs):
        def f(carry, x):
            interal_objs = self.extract(carry)
            updated_objs, output = self.subroutine(internal_objs, x, self.params)
            new_carry = self.compress(updated_objs)
            return new_carry, output

        init_carry = self.compress(objs)
        final_carry, outputs = F.functional_for(f, init_carry, xs)
        final_objs = self.extract(final_carry)
        outputs = Q(outputs, self.output_units)
        if self.measure_var:
            outputs = Normal(outputs, self.measure_var)

        return final_objs, outputs
