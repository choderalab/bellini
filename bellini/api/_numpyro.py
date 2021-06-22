# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini

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
    observed_nodes = [node for node in g.nodes if hasattr(node, "observed")]
    observed_nodes = [node for node in observed_nodes if node.observed is True]

    if len(observed_nodes) != 1:
        raise NotImplementedError("Now we only support one observation of one node.")
    observed_node = observed_nodes[0]

    def model(obs = None):
        model_dict = {}

        def eval(node):
            """ Evaluate node values recursively using DP """
            if node in model_dict.keys():
                return
            elif isinstance(node, bellini.quantity.Quantity):
                #print(node, "param", id(node))
                model_dict[node] = node
            else:
                if isinstance(node, bellini.distributions.SimpleDistribution):
                    if node is observed_node:
                        obs_data = obs
                    else:
                        obs_data = None

                    name = node.name

                    parameters = {}
                    for param_name, param in node.parameters.items():
                        eval(param)
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
                    #print(name, "simple", model_dict[id(node)], id(node))

                elif isinstance(node, bellini.distributions.ComposedDistribution):
                    name = node.name
                    #print(name, "composed", id(node))
                    op = bellini.distributions.OPS[node.op]
                    distributions = node.distributions
                    for param in distributions:
                        eval(param)

                    assert len(distributions) == 2
                    sample = op(
                        model_dict[distributions[0]].magnitude,
                        model_dict[distributions[1]].magnitude,
                    )

                    if getattr(node, "trace", None):
                        numpyro.deterministic(node.name, sample)

                    model_dict[node] = bellini.quantity.Quantity(
                        sample,
                        node.units,
                    )

                elif isinstance(node, bellini.distributions.TransformedDistribution):
                    name = node.name
                    op = bellini.distributions.OPS[node.op]
                    eval(node.distribution)

                    if op == 'pow':
                        sample = op(
                            model_dict[node.distribution.magnitude],
                            node.order
                        )
                    else:
                        sample = op(
                            model_dict[node.distribution.magnitude]
                        )

                    if getattr(node, "trace", None):
                        numpyro.deterministic(node.name, sample)

                    model_dict[node] = bellini.quantity.Quantity(
                        sample,
                        node.units,
                    )

        eval(observed_node)

        return model_dict[observed_node].magnitude, model_dict

    return model
