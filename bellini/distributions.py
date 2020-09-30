# =============================================================================
# IMPORTS
# =============================================================================
import abc
import bellini

# =============================================================================
# BASE CLASSES
# =============================================================================
class Distribution(abc.ABC):
    """ Base class for distributions. """
    def __init__(self, **parameters):
        for name, parameter in parameters.items():
            setattr(self, name, parameter)

class ComposedDistribution(Distribution):
    """ A composed distribution made of two distributions. """
    def __init__(self, distributions, operator):
        super(ComposedDistribution, self).__init__()

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Normal(Distribution):
    """ Normal distribution. """
    def __init__(self, loc, scale):
        super(Normal, self).__init__(loc=loc, scale=scale)
