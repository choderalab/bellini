""" Module containing objects that model physical laws """


import bellini
from bellini.quantity import Quantity
from bellini.distributions import Distribution, _JITDistribution
import numpy as np
from bellini.reference import Reference as Ref


class Law(object):
    """ An object that applies some physical law on a group, grabbing inputs
    dictated by `input_mapping`, applying `law_fn`, and returning a new instance
    with the law applied. """
    def __init__(self, law_fn, input_mapping, output_labels=None, name=None, params=None, group_create_fn=None):
        """
        Parameters
        ----------

        law_fn : Python callable
            a function that takes kwarg inputs based on
            `input_mapping`'s labels and returns a dict storing law outputs. the
            dict should have labels (`bellini.Reference`) corresponding to the attribute the
            result will be stored in when the law is applied to an object, and
            values of those resulting outputs. `fn` should expected Quantity inputs
            and must also return Quantity outputs.

        input_mapping : dict
            a dict mapping `law_fn` kwarg labels (str) to attributes names (str),
            which will be used to retrieve inputs from the given group

        output_labels : list
            output labels of `law_fn`, which are used to connect
            inputs and outputs during compilation. this is not necessary if
            `law_fn` is written purely using `bellini.api.functional calls`,
            which keeps track of transformations automatically, but is required
            if you perform computation using an accelerated framework e.g. jax, torch

        params : dict, optional
            parameters of law_fn that do not rely on inputs

        group_create_fn : Python callable, optional
            a function that takes
            law_fn's outputs and the original `Group` and returns a new `Group`.
            If not provided, the returned `Group` after applying a law will just
            be a copy of the original group, with new attributes set according
            to the output of law_fn and/or output_labels
        """
        # TODO: update output_labels docstring to reflect that you don't need
        # to provide output_labels but that law_fn must return a dict with
        # `Ref` keys you want to use default group creation

        assert isinstance(input_mapping, dict)

        all_str_keys = np.array([isinstance(key, str) for key in input_mapping.keys()]).all()
        all_int_keys = np.array([isinstance(key, int) for key in input_mapping.keys()]).all()
        assert all_str_keys or all_int_keys, ("input_mapping must either be for one"
        " Group specifically (in which all keys should be strings), or be the "
        " same Group across different timesteps (in which all keys should be ints)"
        )

        # if all string keys, we're only looking at the most recent timestep
        if all_str_keys:
            input_mapping = {0: input_mapping}
        self.input_mapping = input_mapping


        if output_labels:
            assert np.array([
                isinstance(label, Ref) for label in output_labels
            ]).all(), "all output_labels must be Reference objects"
        self.output_labels = output_labels
        self.law_fn = law_fn
        self.group_create_fn = group_create_fn
        self.params = params

        if name:
            self.name = name
        else:
            name = f"Law with input mapping {self.input_mapping}, function {law_fn}, and params {params}"

    def __repr__(self):
        return self.name

    def _retrieve_args(self, group_dict):
        def get_ref_in_group(group, ref):
            if isinstance(ref, Ref):
                attr = getattr(group, ref.name)
                return ref.retrieve_index(attr)
            elif isinstance(ref, str):
                return getattr(group, ref)
            elif isinstance(ref, dict):
                dict_arg = {}
                for key, subref in ref.items():
                    dict_arg[key] = get_ref_in_group(group, subref)
                return dict_arg
            else:
                raise ValueError(f"{group} and params does not have required attribute {ref} for use in {self}")

        args = {}
        for timestep, input_mapping in self.input_mapping.items():
            group = group_dict[timestep]
            for fn_kwarg, input_ref in input_mapping.items():
                    args[fn_kwarg] = get_ref_in_group(group, input_ref)

        return args

    def __call__(self, group_dict):

        # so you can call a law on a single group
        if isinstance(group_dict, bellini.Group):
            group_dict = {0: group_dict}

        for group in group_dict.values():
            assert isinstance(group, bellini.Group)

        inputs = self._retrieve_args(group_dict)
        inputs.update(self.params)

        def contains_dist(arg):
            if isinstance(arg, bellini.Distribution):
                return True
            else:
                if isinstance(arg, (list, tuple)):
                    return np.array([contains_dist(r) for r in arg]).any()
                elif isinstance(arg, dict):
                    return np.array([contains_dist(r) for r in arg.values()]).any()
                else:
                    return False

        # compute values
        is_dist = np.array([
            contains_dist(arg) for arg in inputs.values()
        ])
        if is_dist.any():
            #assert self.output_labels is not None
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
                    elif isinstance(arg, Distribution):
                        return bellini.Quantity(arg.magnitude, arg.units)
                    else:
                        return bellini.Quantity(arg)

            deterministic_args = {}
            for key, arg in inputs.items():
                deterministic_args[key] = to_quantity(arg)

            outputs = {}
            deterministic_outputs = self.law_fn(**deterministic_args)
            for label in deterministic_outputs.keys():#self.output_labels:
                outputs[label] = _JITDistribution(
                    self.law_fn,
                    inputs,
                    label,
                    deterministic_outputs=deterministic_outputs
                )
        else:
            outputs = self.law_fn(**inputs)

        latest_group = group_dict[0]

        # create new group with law applied
        if self.group_create_fn:
            new_group = self.group_create_fn(outputs, latest_group, self)
        else:  # default behavior
            new_group = bellini.LawedGroup(latest_group, self)
            for ref, value in outputs.items():
                name = ref.name
                if hasattr(new_group, name):
                    item = getattr(new_group, name)
                    ref.set_index(item, value)
                else:
                    assert ref.is_base(), "can't subindex something that doesn't exist!"
                    setattr(new_group, ref.name, value)

        return new_group
