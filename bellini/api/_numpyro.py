# =============================================================================
# IMPORTS
# =============================================================================
import numpyro
import bellini

def graph_to_numpyro_model(g):
    """ Convert a belief graph to a `numpyro` model. """
    # TODO: `eval` is clearly not the best way to approach this
    # initialize code
    _code = []

    
