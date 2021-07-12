import bellini
from bellini.units import *
from bellini.quantity import Quantity
from bellini.distributions import Distribution, _JITDistribution
import numpy as np

class Law(object):
    def __init__(self, fn, input_mapping, output_labels=None, name=None, params=None):
        """
        An object that applies some physical law on a group, grabbing inputs
        dictated by `input_mapping`, applying `fn`, and returning a new instance
        with the law applied.

        Parameters:
        ----------
        fn: func - a function that takes kwarg inputs based on
            `input_mapping`'s labels and returns a dict storing law outputs. the
            dict should have labels (str) corresponding to the attribute the
            result will be stored in when the law is applied to an object, and
            values of those resulting outputs. `fn` should expected Quantity inputs
            and must also return Quantity outputs.
        input_mapping: dict - a dict mapping `fn` kwarg labels (str) to attributes
            names (str), which will be used to retrieve inputs from the given group
        output_labels: list - output labels of `fn`, which are used to connect
            inputs and outputs during compilation. this is not necessary if
            `fn` is written purely using bellini.api.functional calls, which keeps track of
            transformations automatically, but is required if you perform
            computation using an accelerated framework e.g. jax, torch
        """
        self.input_mapping = input_mapping
        self.output_labels = output_labels
        self.fn = fn
        self.params = params

        if name:
            self.name = name
        else:
            name = f"Law with input mapping {self.input_mapping}, function {fn}, and params {params}"

    def __repr__(self):
        return self.name

    def __call__(self, group):
        assert isinstance(group, bellini.Group)

        # retrieve inputs
        inputs = {}
        for kwarg, inpt in self.input_mapping.items():
            if hasattr(group, inpt):
                inputs[kwarg] = getattr(group, inpt)
            elif hasattr(params, inpt):
                inputs[kwarg] = getattr(params, inpt)
            else:
                raise ValueError(f"{group} and params does not have required attribute {inpt} for use as keyword {kwarg} in {self}")

        # compute values
        is_dist = np.array([
            isinstance(arg, bellini.Distribution) for arg in inputs.values()
        ])
        if is_dist.any():
            assert self.output_labels is not None
            outputs = {}
            for label in self.output_labels:
                outputs[label] = _JITDistribution(
                    self.fn,
                    inputs,
                    label
                )
        else:
            outputs = self.fn(**inputs)

        # create new group with law applied
        new_group = bellini.LawedGroup(group, self)
        for attr, value in outputs.items():
            setattr(new_group, attr, value)

        return new_group
