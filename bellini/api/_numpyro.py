# =============================================================================
# IMPORTS
# =============================================================================
import jax
import jax.numpy as jnp
import numpyro

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _delta(distribution):
    return distribution.x

def _normal(distribution):
    return numpyro.distributions.Normal(
        loc=distribution.loc,
        scale=distribution.scale,
    )
