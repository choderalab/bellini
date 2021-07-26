import bellini
from bellini.quantity import Quantity
from bellini.distributions import Distribution, _JITDistribution
import numpy as np
from bellini.reference import Reference as Ref

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
        if output_labels:
            assert np.array([
                isinstance(label, Ref) for label in output_labels
            ]).all(), "all output_labels must be Reference objects"
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
            # if we want to grab something from an attribute that's a dict
            if isinstance(inpt, (tuple, list)):
                assert len(inpt) == 2, "we only support indexing one layer deep"
                inpt, key = inpt
                dict_attr = getattr(group, inpt, None)
                assert isinstance(dict_attr, dict)
                if key in dict_attr.keys():
                    inputs[kwarg] = dict_attr[key]
                    continue
            elif isinstance(inpt, Ref):
                attr = getattr(group, inpt.name)
                inputs[kwarg] = inpt.retrieve_index(attr)
                continue
            else:
                if hasattr(group, inpt):
                    inputs[kwarg] = getattr(group, inpt)
                    continue
            raise ValueError(f"{group} and params does not have required attribute {inpt} for use as keyword {kwarg} in {self}")

        inputs.update(self.params)

        # compute values
        is_dist = np.array([
            isinstance(arg, bellini.Distribution) for arg in inputs.values()
        ])
        if is_dist.any():
            assert self.output_labels is not None
            # compute deterministic outputs on the outside
            # so we can reduce computation by only running `fn` once
            def to_quantity(arg):
                if isinstance(arg, bellini.Quantity):
                    return arg
                else:
                    if isinstance(arg, (list, tuple)):
                        return [to_quantity(r) for r in arg]
                    elif isinstance(arg, dict):
                        return {key: to_quantity(value) for key, value in arg.items()}
                    else:
                        return bellini.Quantity(arg.magnitude, arg.units)

            deterministic_args = {}
            for key, arg in inputs.items():
                deterministic_args[key] = to_quantity(arg)

            outputs = {}
            deterministic_outputs = self.fn(**deterministic_args)
            for label in self.output_labels:
                outputs[label] = _JITDistribution(
                    self.fn,
                    inputs,
                    label,
                    deterministic_outputs=deterministic_outputs
                )
        else:
            outputs = self.fn(**inputs)

        # create new group with law applied
        new_group = bellini.LawedGroup(group, self)
        for ref, value in outputs.items():
            name = ref.name
            if hasattr(new_group, name):
                item = getattr(new_group, name)
                ref.set_index(item, value)
            else:
                assert ref.is_base(), "can't subindex something that doesn't exist!"
                setattr(new_group, ref, value)

        return new_group
