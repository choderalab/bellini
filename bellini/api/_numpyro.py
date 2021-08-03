# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini
import bellini.api.functional as F
import jax.numpy as jnp
import warnings
from bellini.units import get_internal_units, to_internal_units, ureg

# =============================================================================
# Compilation
# =============================================================================

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
        bellini.set_infer(True)
        model_dict = {} # serves as model trace as well as DP lookup table
        _jit_dist_cache = {} # serves as _jit_dist output cache

        def eval_node(node):
            """ Evaluate node values recursively using DP """
            if node in model_dict.keys():
                return
            elif isinstance(node, bellini.quantity.Quantity):
                model_dict[node] = node.jnp()
            else:
                # draw directly from numpyro function for simple distribution
                if isinstance(node, bellini.distributions.SimpleDistribution):
                    name = node.name
                    obs_data = None

                    if node in observed_nodes:
                        if obs is not None and node in obs.keys():
                            obs_data = obs[node].to(node.units).magnitude
                        else:
                            warnings.warn(f"observed node {name} was not given data to condition on. no conditioning performed.")

                    parameters = {}
                    for param_name, param in node.parameters.items():
                        eval_node(param)
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
                        eval_node(param)

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
                    op = getattr(jnp, node.op)
                    for arg in node.args:
                        eval_node(arg)

                    jax_args = [
                        to_internal_units(model_dict[arg]).magnitude
                        for arg in node.args
                    ]

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
                                eval_node(arg_val)
                        else:
                            eval_node(arg)

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
                    eval_node(dist)

                    sample = (model_dict[dist] * dist.scaling_factor).magnitude

                    if getattr(node, "trace", None):
                        numpyro.deterministic(node.name, sample)

                    model_dict[node] = bellini.quantity.Quantity(
                        sample,
                        node.units,
                    )

        for node in observed_nodes:
            eval_node(node)

        bellini.set_infer(False)
        return model_dict

    return model
