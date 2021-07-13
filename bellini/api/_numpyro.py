# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini
import bellini.api.functional as F
import jax.numpy as jnp
from bellini.units import get_internal_units, to_internal_units, ureg

# =============================================================================
# Compilation
# =============================================================================

def graph_to_numpyro_model(g):
    """ Convert a belief graph to a `numpyro` model.

    The current design involves removing units from parameters, sampling from a
    numpyro distribution to get a dimensionless quantity, then reapplying units
    when the sample is being tracked in model_dict. This is because numpyro uses
    unitless quantities when computing log probs, which leads to dimensionality
    errors if there are mixed unitless / unit quantities. For similar reasons,
    the final observed node and observed value must be unitless as well, but
    units can be recovered from `model_dict` which is returned, as well as from
    the computation graph.

    TODO: is there an easy way to retain units for the observed node? It would
    make for a more intuitive interface.

    note: doesn't actually use the explicit nx graph, instead uses implicit
    param graph? need to think about design
    """
    import networkx as nx
    observed_nodes = [node for node in g.nodes if getattr(node, "observed", None)]

    if len(observed_nodes) != 1:
        raise NotImplementedError("Now we only support one observation of one node.")
    observed_node = observed_nodes[0]

    def model(obs = None):
        bellini.set_infer(True)
        model_dict = {}

        def eval_node(node):
            """ Evaluate node values recursively using DP """
            if node in model_dict.keys():
                return
            elif isinstance(node, bellini.quantity.Quantity):
                model_dict[node] = node.jnp()
            else:
                if isinstance(node, bellini.distributions.SimpleDistribution):
                    if node is observed_node:
                        obs_data = obs
                    else:
                        obs_data = None

                    name = node.name

                    parameters = {}
                    for param_name, param in node.parameters.items():
                        eval_node(param)
                        parameters[param_name] = model_dict[param].magnitude

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

                    if getattr(node, "trace", None):
                        numpyro.deterministic(node.name, sample)

                    model_dict[node] = bellini.quantity.Quantity(
                        sample,
                        node.internal_units,
                    ).to(node.units)


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
                        node.internal_units,
                    ).to(node.units)

                elif isinstance(node, bellini.distributions._JITDistribution):
                    name = node.name
                    fn = node.fn
                    inputs = node.inputs
                    label = node.label

                    sampled_inputs = {
                        key: model_dict[arg]
                        for key, arg in inputs.items()
                    }

                    sample = fn(**sampled_inputs)[label]

                    if getattr(node, "trace", None):
                        numpyro.deterministic(node.name, sample)

                    model_dict[node] = bellini.quantity.Quantity(
                        sample,
                        node.internal_units,
                    ).to(node.units)

        eval_node(observed_node)

        bellini.set_infer(False)
        return model_dict[observed_node].magnitude, model_dict

    return model
