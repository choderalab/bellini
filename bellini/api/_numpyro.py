# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini

def graph_to_numpyro_model(g):
    """ Convert a belief graph to a `numpyro` model. """
    import networkx as nx
    observed_nodes = [node for node in g.nodes if node.observed is True]

    if len(observed_nodes) != 1:
        raise NotImplementedError("Now we only support one type of observation.")

    edges = list(nx.bfs_edges(g, source=observed_nodes, reverse=True))[::-1]
    nodes = [edge[0] for edge in edges]
    print(nodes)
