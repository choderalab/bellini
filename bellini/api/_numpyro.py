# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini

def graph_to_numpyro_model(g):
    """ Convert a belief graph to a `numpyro` model. """
    import networkx as nx
    observed_nodes = [node for node in g.nodes if hasattr(node, "observed")]
    observed_nodes = [node for node in observed_nodes if node.observed is True]

    if len(observed_nodes) != 1:
        raise NotImplementedError("Now we only support one type of observation.")
    observed_node = observed_nodes[0]

    edges = list(nx.bfs_edges(g, source=observed_node, reverse=True))[::-1]
    nodes = [edge[1] for edge in edges]

    def model():
        for node in nodes:

            if isinstance(node, bellini.distributions.SimpleDistribution):
                name = node.name

                def _apply_parameter(parameter):
                    if isinstance(parameter, bellini.quantity.Quantity):
                        return parameter.magnitude
                    elif isinstance(parameter, bellini.distributions.Distribution):
                        return locals()[parameter.name]

                parameters = [_apply_parameter(param) for param_name, param in node.parameters.items()]

                locals()[name] = numpyro.sample(
                    name,
                    getattr(
                        numpyro.distributions,
                        node.__class__.__name__,
                    )(
                        *parameters,
                    )
                )

            if isinstance(model, bellini.distributions.ComposedDistribution):
                name = str(node)
                op = bellini.distributions.OPS[node.op]
                distributions = node.distributions
                assert len(distributions) == 2
                locals()[name] = op(
                    distributions[0],
                    distributions[1],
                )

    return model
