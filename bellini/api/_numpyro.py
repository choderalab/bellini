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
                name = str(node)
                locals()[name] = numpyro.sample(
                    name,
                    getattr(
                        numpyro.distributions,
                        node.__class__.__name__,
                    )(
                        *[param._value for param in node.parameters],
                    )
                )

            if isinstance(model, bellini.distributions.ComposedDistribution):
                name = str(node)
                op = node.op
                distributions = node.distributions
                assert len(distributions) == 2
