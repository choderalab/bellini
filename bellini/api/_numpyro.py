# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini
import bellini.api.functional as F
from bellini.api import utils
import jax
import jax.numpy as jnp
import warnings
from bellini.units import get_internal_units, to_internal_units, ureg

# =============================================================================
# Compilation
# =============================================================================

def _eval_node(node, model_dict, _jit_dist_cache, obs=None):
    """ Evaluate node values recursively using DP """
    if node is None: # allows None nodes
        return None
    elif isinstance(node, (list, tuple)):
        return [_eval_node(subnode, model_dict, _jit_dist_cache, obs) for subnode in node]
    elif node in model_dict.keys():
        pass
    elif callable(node):  # e.g. a function
        model_dict[node] = node
    elif isinstance(node, bellini.quantity.Quantity):
        model_dict[node] = node.jnp()
    else:
        # draw directly from numpyro function for simple distribution
        if isinstance(node, bellini.distributions.SimpleDistribution):
            name = node.name
            obs_data = None

            if obs is not None and node in obs.keys():
                obs_data = obs[node].to(node.units).magnitude
            elif getattr(node, "observed", False):
                warnings.warn(f"observed node {name} was not given data to condition on. no conditioning performed.")

            parameters = {}
            for param_name, param in node.parameters.items():
                _eval_node(param, model_dict, _jit_dist_cache, obs)
                # we assume all the parameters are of the same unit as node
                # i think this should be reasonable?
                parameters[param_name] = model_dict[param].to(node.units).magnitude

            sample = numpyro.sample(
                name,
                getattr(
                    numpyro.distributions,
                    node.__class__.__name__,
                )(
                    **parameters,
                ),
                obs = obs_data
            )

            model_dict[node] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

        # compute composed distribution based on parameters
        elif isinstance(node, bellini.distributions.ComposedDistribution):
            name = node.name
            op = getattr(F, node.op)
            distributions = node.distributions
            for param in distributions:
                _eval_node(param, model_dict, _jit_dist_cache, obs)

            assert len(distributions) == 2
            sample = op(
                to_internal_units(model_dict[distributions[0]]).magnitude,
                to_internal_units(model_dict[distributions[1]]).magnitude,
            )

            dist1_int_units = get_internal_units(model_dict[distributions[0]])
            dist2_int_units = get_internal_units(model_dict[distributions[1]])
            if node.op in ("add", "sub"):
                sample_units = dist1_int_units
            else:
                sample_units = op(
                    dist1_int_units,
                    dist2_int_units
                )

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[node] = bellini.quantity.Quantity(
                sample,
                sample_units,
            ).to(node.units)

        # compute transposed distribution based on parameters
        elif isinstance(node, bellini.distributions.TransformedDistribution):
            name = node.name
            op = getattr(F, node.op)
            for arg in node.args:
                _eval_node(arg, model_dict, _jit_dist_cache, obs)

            def _to_internal_units_mag(arg):
                return to_internal_units(model_dict[arg]).magnitude

            args_to_jax = utils._to_x_constructor(_to_internal_units_mag)
            jax_args = args_to_jax(node.args)

            sample = op(*jax_args, **node.kwargs)

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[node] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

        # compute fn outputs based on parameters, with caching
        elif isinstance(node, bellini.distributions._JITDistribution):
            name = node.name
            fn = node.fn
            inputs = node.inputs
            label = node.label

            def _gen_cache_key(inputs):
                hashable_inputs = []
                for key, entry in inputs.items():
                    if isinstance(entry, dict):
                        hashable_inputs.append(
                            (
                                key,
                                tuple(entry.items())
                            )
                        )
                    else:
                        hashable_inputs.append((key, entry))
                return tuple(hashable_inputs)

            for arg in inputs.values():
                if isinstance(arg, dict):
                    for arg_val in arg.values():
                        _eval_node(arg_val, model_dict, _jit_dist_cache, obs)
                elif isinstance(arg, (list, tuple)):
                    model_dict[node] = [_eval_node(subnode, model_dict, _jit_dist_cache, obs) for subnode in node]
                else:
                    _eval_node(arg, model_dict, _jit_dist_cache, obs)

            # caching fn outputs since fn could potentially be expensive
            _cache_key = (fn, _gen_cache_key(inputs))
            if _cache_key not in _jit_dist_cache.keys():
                sampled_inputs = {}
                for key, arg in inputs.items():
                    if isinstance(arg, dict):
                        sampled_inputs[key] = {
                            k: model_dict[v]
                            for k,v in arg.items()
                        }
                    elif arg is None:  # for args that are None
                        sampled_inputs[key] = None
                    elif isinstance(arg, list):
                        sampled_inputs[key] = [model_dict[a] for a in arg]
                    elif isinstance(arg, tuple):
                        sampled_inputs[key] = tuple([model_dict[a] for a in arg])
                    else:
                        sampled_inputs[key] = model_dict[arg]

                _jit_dist_cache[_cache_key] = fn(**sampled_inputs)

            sample = _jit_dist_cache[_cache_key][label]

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample.magnitude)

            model_dict[node] = sample.to(node.units)

        # compute unit scaling
        elif isinstance(node, bellini.distributions.UnitChangedDistribution):
            name = node.name
            dist = node.distribution
            _eval_node(dist, model_dict, _jit_dist_cache, obs)

            sample = (model_dict[dist] * dist.scaling_factor).magnitude

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[node] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

    return model_dict[node]

def _lax_eval_node(node, model_dict, _jit_dist_cache, id_to_node={}, obs=None):
    """ Evaluate node values recursively using DP """

    def eval_unvisited_node(node, model_dict, _jit_dist_cache, id_to_node={}, obs=None):
        # draw directly from numpyro function for simple distribution
        if isinstance(node, bellini.distributions.SimpleDistribution):
            name = node.name
            obs_data = None

            if obs is not None and node in obs.keys():
                obs_data = obs[node].to(node.units).magnitude
            elif getattr(node, "observed", False):
                warnings.warn(f"observed node {name} was not given data to condition on. no conditioning performed.")

            parameters = {}
            for param_name, param in node.parameters.items():
                _lax_eval_node(param, model_dict, _jit_dist_cache, id_to_node, obs)
                # we assume all the parameters are of the same unit as node
                # i think this should be reasonable?
                parameters[param_name] = model_dict[id(param)].to(node.units).magnitude

            sample = numpyro.sample(
                name,
                getattr(
                    numpyro.distributions,
                    node.__class__.__name__,
                )(
                    **parameters,
                ),
                obs = obs_data
            )
            print("lax sample!", name)

            model_dict[id(node)] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

        # compute composed distribution based on parameters
        elif isinstance(node, bellini.distributions.ComposedDistribution):
            name = node.name
            op = getattr(F, node.op)
            distributions = node.distributions
            for param in distributions:
                _lax_eval_node(param, model_dict, _jit_dist_cache, id_to_node, obs)

            assert len(distributions) == 2
            sample = op(
                to_internal_units(model_dict[id(distributions[0])]).magnitude,
                to_internal_units(model_dict[id(distributions[1])]).magnitude,
            )

            dist1_int_units = get_internal_units(model_dict[id(distributions[0])])
            dist2_int_units = get_internal_units(model_dict[id(distributions[1])])
            if node.op in ("add", "sub"):
                sample_units = dist1_int_units
            else:
                sample_units = op(
                    dist1_int_units,
                    dist2_int_units
                )

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[id(node)] = bellini.quantity.Quantity(
                sample,
                sample_units,
            ).to(node.units)

        # compute transposed distribution based on parameters
        elif isinstance(node, bellini.distributions.TransformedDistribution):
            name = node.name
            op = getattr(F, node.op)
            for arg in node.args:
                _lax_eval_node(arg, model_dict, _jit_dist_cache, id_to_node, obs)

            jax_args = [
                to_internal_units(model_dict[id(arg)]).magnitude
                for arg in node.args
            ]

            sample = op(*jax_args, **node.kwargs)

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[id(node)] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

        # compute fn outputs based on parameters, with caching
        elif isinstance(node, bellini.distributions._JITDistribution):
            name = node.name
            fn = node.fn
            inputs = node.inputs
            label = node.label

            def _gen_cache_key(inputs):
                hashable_inputs = []
                for key, entry in inputs.items():
                    if isinstance(entry, dict):
                        hashable_inputs.append(
                            (
                                key,
                                tuple(entry.items())
                            )
                        )
                    else:
                        hashable_inputs.append((key, entry))
                return tuple(hashable_inputs)

            for arg in inputs.values():
                if isinstance(arg, dict):
                    for arg_val in arg.values():
                        _lax_eval_node(arg_val, model_dict, _jit_dist_cache, id_to_node, obs)
                else:
                    _lax_eval_node(arg, model_dict, _jit_dist_cache, id_to_node, obs)

            # caching fn outputs since fn could potentially be expensive
            _cache_key = (fn, _gen_cache_key(inputs))
            if _cache_key not in _jit_dist_cache.keys():
                sampled_inputs = {}
                for key, arg in inputs.items():
                    if isinstance(arg, dict):
                        sampled_inputs[key] = {
                            k: model_dict[id(v)]
                            for k,v in arg.items()
                        }
                    else:
                        sampled_inputs[key] = model_dict[id(arg)]

                _jit_dist_cache[_cache_key] = fn(**sampled_inputs)

            sample = _jit_dist_cache[_cache_key][label]

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample.magnitude)

            model_dict[id(node)] = sample.to(node.units)

        # compute unit scaling
        elif isinstance(node, bellini.distributions.UnitChangedDistribution):
            name = node.name
            dist = node.distribution
            _lax_eval_node(dist, model_dict, _jit_dist_cache, id_to_node, obs)

            sample = (model_dict[id(dist)] * dist.scaling_factor).magnitude

            if getattr(node, "trace", None):
                numpyro.deterministic(node.name, sample)

            model_dict[id(node)] = bellini.quantity.Quantity(
                sample,
                node.units,
            )

    node_id = id(node)
    #print(node)
    is_quant = isinstance(node, bellini.quantity.Quantity)
    if is_quant:  # filter out Quantities first to prevent tracer issues
        #print(type(node))
        model_dict[node_id] = node.jnp()
    else:
        already_evaled = (node_id in model_dict.keys())
        if already_evaled:
            pass
        else:
            eval_unvisited_node(node, model_dict, _jit_dist_cache, id_to_node, obs)

    #print(model_dict[node_id])
    return model_dict[node_id]


def _compile(out_node):
    """ Compile an unobserved node's computation graph """
    assert not getattr(out_node, "observed", False), ("no way of observing"
        " this node using _compile")

    with bellini.inference():
        model_dict = {} # serves as model trace as well as DP lookup table
        _jit_dist_cache = {} # serves as _jit_dist output cache
        return _lax_eval_node(out_node, model_dict, _jit_dist_cache, obs=None)


def graphs_to_numpyro_model(graph_list):
    """ Convert a belief graph to a `numpyro` model.

    The current design involves removing units from parameters, sampling from a
    numpyro distribution to get a dimensionless quantity, then reapplying units
    when the sample is being tracked in model_dict. This is because numpyro uses
    unitless quantities when computing log probs, which leads to dimensionality
    errors if there are mixed unitless / unit quantities.

    note: doesn't actually use the explicit nx graph, instead uses implicit
    param graph? need to think about design
    """
    import networkx as nx
    G = nx.compose_all(graph_list)
    observed_nodes = [node for node in G.nodes if getattr(node, "observed", None)]

    def model(obs = None):
        with bellini.inference():
            model_dict = {} # serves as model trace as well as DP lookup table
            _jit_dist_cache = {} # serves as _jit_dist output cache
            for node in observed_nodes:
                _eval_node(node, model_dict, _jit_dist_cache, obs)

        return model_dict

    return model
